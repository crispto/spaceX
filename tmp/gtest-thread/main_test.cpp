#include <gtest/gtest.h>
TEST(META, Info)
{
    printf("%s", "test env set ok");
}
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
