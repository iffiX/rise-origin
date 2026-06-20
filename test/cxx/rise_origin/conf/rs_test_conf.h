#ifndef RS_TEST_CONF_H
#define RS_TEST_CONF_H

#include "rise_origin/device/platform/rs_platform.h"

/*****************************************************************
 * Device backend
 *****************************************************************/
// The backend is selected by cmake through RISE_ORIGIN_TEST_BACKEND, which
// injects RS_TEST_DEVICE_BACKEND_CUDA for the cuda backend. The cpu backend is
// the default when nothing is injected.
#if !defined(RS_TEST_DEVICE_BACKEND_CPU) && !defined(RS_TEST_DEVICE_BACKEND_CUDA)
#define RS_TEST_DEVICE_BACKEND_CPU
#endif

#if defined(RS_TEST_DEVICE_BACKEND_CUDA)
#define RS_TEST_SETUP_DEVICE_PROVIDER() provider.init({RSE_DEVICE_TYPE_CUDA, 0})
#define RS_TEST_DEVICE_PROVIDER_TYPE RS_MemoryCudaProvider
#else
#define RS_TEST_SETUP_DEVICE_PROVIDER() provider.init(0)
#define RS_TEST_DEVICE_PROVIDER_TYPE RS_MemoryCPUProvider
#endif

#endif  // RS_TEST_CONF_H
