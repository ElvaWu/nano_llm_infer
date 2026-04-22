#pragma once

#include <cmath>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>

namespace test_util {

struct TestState {
    int total = 0;
    int failed = 0;
};

inline TestState& state() {
    static TestState s;
    return s;
}

inline void fail(const std::string& expr, const std::string& file, int line, const std::string& msg = "") {
    TestState& s = state();
    s.failed++;
    std::cerr << "[FAIL] " << file << ":" << line << " - " << expr;
    if (!msg.empty()) {
        std::cerr << " (" << msg << ")";
    }
    std::cerr << "\n";
}

inline void pass_or_fail(bool ok, const std::string& expr, const std::string& file, int line, const std::string& msg = "") {
    TestState& s = state();
    s.total++;
    if (!ok) {
        fail(expr, file, line, msg);
    }
}

inline int finalize() {
    const TestState& s = state();
    if (s.failed == 0) {
        std::cerr << "[PASS] " << s.total << " checks\n";
        return 0;
    }
    std::cerr << "[FAIL] " << s.failed << " / " << s.total << " checks failed\n";
    return 1;
}

}  // namespace test_util

#define EXPECT_TRUE(expr) \
    ::test_util::pass_or_fail(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

#define EXPECT_EQ(lhs, rhs)                                                                        \
    do {                                                                                            \
        const auto _lhs = (lhs);                                                                   \
        const auto _rhs = (rhs);                                                                   \
        ::test_util::pass_or_fail(_lhs == _rhs, #lhs " == " #rhs, __FILE__, __LINE__);           \
    } while (false)

#define EXPECT_NEAR(lhs, rhs, eps)                                                                 \
    do {                                                                                            \
        const double _lhs = static_cast<double>(lhs);                                              \
        const double _rhs = static_cast<double>(rhs);                                              \
        const double _eps = static_cast<double>(eps);                                              \
        ::test_util::pass_or_fail(std::fabs(_lhs - _rhs) <= _eps,                                  \
                                  #lhs " ~= " #rhs, __FILE__, __LINE__);                           \
    } while (false)

#define EXPECT_THROW(stmt)                                                                          \
    do {                                                                                             \
        bool _thrown = false;                                                                        \
        try {                                                                                        \
            stmt;                                                                                    \
        } catch (const std::exception&) {                                                           \
            _thrown = true;                                                                          \
        } catch (...) {                                                                              \
            _thrown = true;                                                                          \
        }                                                                                            \
        ::test_util::pass_or_fail(_thrown, "EXPECT_THROW(" #stmt ")", __FILE__, __LINE__);      \
    } while (false)
