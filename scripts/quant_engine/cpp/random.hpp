// Portable pseudo-random number generation for the Monte Carlo engines.
//
// The standard library distributions (for example std::normal_distribution)
// are implementation defined, so the same seed produces different numbers
// with GCC, Clang and MSVC. To keep simulations reproducible across
// compilers, uniforms come from xoshiro256** (Blackman and Vigna, 2021;
// public-domain reference code) seeded through SplitMix64, and normals are
// produced with the Box-Muller transform.
#pragma once

#include <cmath>
#include <cstdint>

namespace qe {

inline std::uint64_t splitmix64(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// xoshiro256** generator. Independent streams are obtained by hashing the
// pair (seed, stream) with SplitMix64, so every block of Monte Carlo paths
// has its own generator and results do not depend on the number of threads.
class Xoshiro256 {
public:
    explicit Xoshiro256(std::uint64_t seed, std::uint64_t stream = 0) {
        std::uint64_t sm = seed;
        const std::uint64_t base = splitmix64(sm);
        sm = base ^ (0xD1B54A32D192ED03ULL * (stream + 1));
        for (auto& word : s_) word = splitmix64(sm);
    }

    std::uint64_t next() {
        const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Uniform variate on the open interval (0, 1) with 53 random bits.
    double uniform() { return (static_cast<double>(next() >> 11) + 0.5) * 0x1.0p-53; }

    // Standard normal variate (Box-Muller, both values of each pair are used).
    double normal() {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        const double radius = std::sqrt(-2.0 * std::log(uniform()));
        const double angle = 6.283185307179586476925 * uniform();
        spare_ = radius * std::sin(angle);
        has_spare_ = true;
        return radius * std::cos(angle);
    }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

    std::uint64_t s_[4];
    bool has_spare_ = false;
    double spare_ = 0.0;
};

}  // namespace qe
