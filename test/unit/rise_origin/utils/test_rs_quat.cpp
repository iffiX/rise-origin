#include <gtest/gtest.h>

#include "rise_origin/utils/rs_quat.h"

using namespace rise;

const float RELATIVE_TOLERANCE_QUAT = 1e-4f;

// Test the constructor and getter
TEST(RS_Quat3DMemTest, ConstructorAndGetters) {
    RS_Quat3DMem<RSE_FLOAT_32> quat(0.574259f, 0.719324f, 0.364164f, 0.142171f);
    float x, y, z, w;
    quat.get(x, y, z, w);

    EXPECT_NEAR(x, 0.574259f, RELATIVE_TOLERANCE_QUAT);
    EXPECT_NEAR(y, 0.719324f, RELATIVE_TOLERANCE_QUAT);
    EXPECT_NEAR(z, 0.364164f, RELATIVE_TOLERANCE_QUAT);
    EXPECT_NEAR(w, 0.142171f, RELATIVE_TOLERANCE_QUAT);
}

// Test the assignment operator
TEST(RS_Quat3DMemTest, AssignmentOperator) {
    RS_Quat3DMem<RSE_FLOAT_32> quat1(0.574259f, 0.719324f, 0.364164f, 0.142171f);
    RS_Quat3DMem<RSE_FLOAT_32> quat2;
    quat2 = quat1;

    float x, y, z, w;
    quat2.get(x, y, z, w);

    EXPECT_NEAR(x, 0.574259f, RELATIVE_TOLERANCE_QUAT);
    EXPECT_NEAR(y, 0.719324f, RELATIVE_TOLERANCE_QUAT);
    EXPECT_NEAR(z, 0.364164f, RELATIVE_TOLERANCE_QUAT);
    EXPECT_NEAR(w, 0.142171f, RELATIVE_TOLERANCE_QUAT);
}
