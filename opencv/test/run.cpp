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

class MathOperationsTest : public ::testing::TestWithParam< ::testing::tuple<int, bool> > {
protected:
    MathOperations math;

    bool checkData() {
        int n = ::testing::get<0>(GetParam());
        bool expectedResult = ::testing::get<1>(GetParam());

        return math.isEven(n) == expectedResult;
    }
};

TEST_P(MathOperationsTest, TestIsEven) {
    EXPECT_TRUE(checkData());
}

INSTANTIATE_TEST_CASE_P(TestMathOperations, MathOperationsTest, ::testing::Combine(::testing::Values(2, 4, 6), ::testing::Bool()));
