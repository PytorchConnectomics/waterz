#pragma once

#include <cstdint>
#include <vector>

/**
 * Weighted union-find with path compression and union by size.
 *
 * Ported from zwatershed's zi/disjoint_sets, simplified for the
 * merge_segments use case.
 */
class DisjointSets {
public:
    explicit DisjointSets(std::size_t n)
        : parent_(n), size_(n, 0)
    {
        for (std::size_t i = 0; i < n; ++i)
            parent_[i] = i;
    }

    void set_size(std::size_t id, std::size_t sz) {
        size_[id] = sz;
    }

    std::size_t find(std::size_t x) {
        std::size_t r = x;
        while (parent_[r] != r)
            r = parent_[r];
        // Path compression
        while (parent_[x] != r) {
            std::size_t nxt = parent_[x];
            parent_[x] = r;
            x = nxt;
        }
        return r;
    }

    /**
     * Merge sets containing a and b.  Returns root of merged set.
     * The larger set becomes the root.
     */
    std::size_t join(std::size_t a, std::size_t b) {
        a = find(a);
        b = find(b);
        if (a == b) return a;
        // Union by size
        if (size_[a] < size_[b])
            std::swap(a, b);
        parent_[b] = a;
        size_[a] += size_[b];
        size_[b] = 0;
        return a;
    }

    std::size_t size_of(std::size_t x) {
        return size_[find(x)];
    }

    std::size_t count() const { return parent_.size(); }

private:
    std::vector<std::size_t> parent_;
    std::vector<std::size_t> size_;
};
