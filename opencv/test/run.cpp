#include "gtest/gtest.h"
#include <gtest/gtest.h>

class MathOperations {
public:
    bool isEven(int n) {
        return n % 2 == 0;
    }

    bool isPositive(int n) {
        return n > 0;
    }
};

// 一个继承了 TestWithPram 的类 test fixture
class MathOperationsTest : public ::testing::TestWithParam< ::testing::tuple<int, bool> > {
protected:
    MathOperations math;

    bool checkData() {
        int n = ::testing::get<0>(GetParam());
        bool expectedResult = ::testing::get<1>(GetParam());

        return math.isEven(n) == expectedResult;
    }
};

// 确定了一个测试束  parameterized test case
TEST_P(MathOperationsTest, TestIsEven) {
    EXPECT_TRUE(checkData());
}

// gtest 自动通过参数来生成相应的testcase， 不用每一个都用TEST 来写
// INSTANTIATE_TEST_CASE_P(banana, MathOperationsTest, ::testing::Combine(::testing::Values(2, 4, 6), ::testing::Values(true)));

INSTANTIATE_TEST_CASE_P(orange, MathOperationsTest, ::testing::Combine(::testing::Values(1, 4, 8, 3), ::testing::Values(false)));
