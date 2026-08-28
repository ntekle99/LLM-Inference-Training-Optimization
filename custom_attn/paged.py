"""Paged KV cache: fixed-size blocks, a block table, and a free-block allocator.

The contiguous cache preallocates ``max_batch_size x max_seq_len`` and wastes
``max_seq_len - actual_len`` per sequence, permanently, for every sequence in the
batch. Paging removes that slack: memory is handed out a block at a time as a
sequence grows, so the cache costs what the sequences actually use rather than
what they might use.

The allocator lives in Python on purpose. It runs once per decode step, not once
per token per head, so it is nowhere near the hot path -- the part that had to be
in CUDA is the gather through the block table, which is in the kernel. Keeping it
here makes refcounting and copy-on-write straightforward to test directly.
"""

import torch

BLOCK_SIZE = 16  # default; kernel supports 16, 32, 64, 128
SUPPORTED_BLOCK_SIZES = (16, 32, 64, 128)


class OutOfBlocks(RuntimeError):
    """The pool is exhausted. Callers are expected to evict or refuse work."""


class PagedKVCache:
    """A pool of fixed-size KV blocks plus per-sequence block tables.

    k_pool/v_pool are [num_blocks, n_kv_heads, BLOCK_SIZE, head_dim]. That layout
    keeps the BLOCK_SIZE tokens of one kv_head contiguous, so the kernel reading a
    tile of consecutive tokens still gets a coalesced load after the indirection.
    """

    def __init__(self, num_blocks, n_kv_heads, head_dim, block_size=BLOCK_SIZE,
                 device="cuda", dtype=torch.float16, allocate_pool=True):
        self.num_blocks = num_blocks
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        if block_size not in SUPPORTED_BLOCK_SIZES:
            raise ValueError(f"block_size must be one of {SUPPORTED_BLOCK_SIZES}")
        self.block_size = block_size
        self.device = device
        self.dtype = dtype

        # allocate_pool=False exercises the bookkeeping without materialising the
        # tensors, so a capacity study over a multi-GiB budget does not need
        # multi-GiB of host RAM to answer a question about block counts.
        shape = (num_blocks, n_kv_heads, block_size, head_dim)
        self._shape = shape
        if allocate_pool:
            self.k_pool = torch.zeros(shape, device=device, dtype=dtype)
            self.v_pool = torch.zeros(shape, device=device, dtype=dtype)
        else:
            self.k_pool = self.v_pool = None

        # Free list as a stack: pop/push are O(1) and recently freed blocks come
        # back first, which keeps the working set warm.
        self._free = list(reversed(range(num_blocks)))
        self._refcount = [0] * num_blocks

        self._blocks = {}   # seq_id -> [physical block ids]
        self._lens = {}     # seq_id -> token count

    # ---------------------------------------------------------------- stats

    @property
    def num_free_blocks(self):
        return len(self._free)

    @property
    def num_used_blocks(self):
        return self.num_blocks - len(self._free)

    def pool_bytes(self):
        n = 1
        for d in self._shape:
            n *= d
        return n * torch.tensor([], dtype=self.dtype).element_size() * 2

    def refcount(self, physical_block):
        return self._refcount[physical_block]

    def blocks_of(self, seq_id):
        return list(self._blocks[seq_id])

    def seq_len(self, seq_id):
        return self._lens[seq_id]

    # ------------------------------------------------------------ allocation

    def _alloc_block(self):
        if not self._free:
            raise OutOfBlocks(
                f"KV pool exhausted ({self.num_blocks} blocks, "
                f"{self.num_used_blocks} in use)")
        b = self._free.pop()
        self._refcount[b] = 1
        return b

    def _release_block(self, b):
        self._refcount[b] -= 1
        if self._refcount[b] == 0:
            self._free.append(b)
        elif self._refcount[b] < 0:
            raise AssertionError(f"block {b} refcount went negative")

    def allocate(self, seq_id, n_tokens=0):
        """Create a sequence with room for n_tokens (contents undefined)."""
        if seq_id in self._blocks:
            raise KeyError(f"sequence {seq_id} already exists")
        n_blocks = (n_tokens + self.block_size - 1) // self.block_size
        blocks = []
        try:
            for _ in range(n_blocks):
                blocks.append(self._alloc_block())
        except OutOfBlocks:
            for b in blocks:          # do not leak a partial allocation
                self._release_block(b)
            raise
        self._blocks[seq_id] = blocks
        self._lens[seq_id] = n_tokens
        return blocks

    def free(self, seq_id):
        for b in self._blocks.pop(seq_id):
            self._release_block(b)
        del self._lens[seq_id]

    # --------------------------------------------------------- prefix sharing

    def fork(self, seq_id, new_seq_id):
        """Share seq_id's blocks with new_seq_id instead of copying them.

        Two sequences with a common prompt prefix then cost one copy of that
        prefix. Divergence is handled by copy-on-write in write_token: the first
        write into a block with refcount > 1 copies it first.
        """
        if new_seq_id in self._blocks:
            raise KeyError(f"sequence {new_seq_id} already exists")
        blocks = list(self._blocks[seq_id])
        for b in blocks:
            self._refcount[b] += 1
        self._blocks[new_seq_id] = blocks
        self._lens[new_seq_id] = self._lens[seq_id]
        return blocks

    def _cow(self, seq_id, slot):
        """Give this sequence a private copy of the block in `slot`."""
        old = self._blocks[seq_id][slot]
        if self._refcount[old] == 1:
            return old
        new = self._alloc_block()
        self.k_pool[new].copy_(self.k_pool[old])
        self.v_pool[new].copy_(self.v_pool[old])
        self._blocks[seq_id][slot] = new
        self._refcount[old] -= 1
        return new

    # ------------------------------------------------------------- appending

    def write_token(self, seq_id, k, v):
        """Append one token's K and V, each [n_kv_heads, head_dim]."""
        pos = self._lens[seq_id]
        slot, off = divmod(pos, self.block_size)

        if slot == len(self._blocks[seq_id]):
            self._blocks[seq_id].append(self._alloc_block())
        phys = self._cow(seq_id, slot)   # no-op unless the block is shared

        self.k_pool[phys, :, off, :] = k
        self.v_pool[phys, :, off, :] = v
        self._lens[seq_id] = pos + 1
        return phys

    def write_prefix(self, seq_id, k, v):
        """Bulk-write a prompt. k, v are [n_tokens, n_kv_heads, head_dim].

        Writes a block at a time rather than a token at a time: a 4096-token
        prompt is 256 slice copies instead of 4096 round trips through Python.
        """
        n = k.shape[0]
        written = 0
        while written < n:
            pos = self._lens[seq_id]
            slot, off = divmod(pos, self.block_size)
            if slot == len(self._blocks[seq_id]):
                self._blocks[seq_id].append(self._alloc_block())
            phys = self._cow(seq_id, slot)

            take = min(self.block_size - off, n - written)
            chunk = slice(written, written + take)
            # [take, Hkv, D] -> [Hkv, take, D] to match the pool layout.
            self.k_pool[phys, :, off:off + take, :] = k[chunk].transpose(0, 1)
            self.v_pool[phys, :, off:off + take, :] = v[chunk].transpose(0, 1)

            self._lens[seq_id] = pos + take
            written += take

    # ----------------------------------------------------------- block table

    def build_block_table(self, seq_ids):
        """Block table [B, max_blocks] int32 and lengths [B] int32, for the kernel.

        Padding entries are never dereferenced: the kernel bounds its token loop
        by lens[b], so slots past the sequence end are not read.
        """
        lens = [self._lens[s] for s in seq_ids]
        max_blocks = max(
            (len(self._blocks[s]) for s in seq_ids), default=0) or 1
        table = torch.zeros((len(seq_ids), max_blocks), dtype=torch.int32)
        for i, s in enumerate(seq_ids):
            blocks = self._blocks[s]
            if blocks:
                table[i, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        return (table.to(self.device),
                torch.tensor(lens, dtype=torch.int32, device=self.device))


__all__ = ["PagedKVCache", "OutOfBlocks", "BLOCK_SIZE", "SUPPORTED_BLOCK_SIZES"]
