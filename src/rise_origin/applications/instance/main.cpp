// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <fmt/format.h>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <iostream>

#include "rise_origin/rise_origin.h"
#include "rise_origin/utils/time/rs_timer.h"

namespace fs = boost::filesystem;
namespace po = boost::program_options;
using namespace std;
using namespace fmt;
using namespace rise;

#define APP_DESCRIPTION                     \
    "RiseOrigin Simulator\n\n"              \
    "Typical Usage:\n"                      \
    "  rise_origin_main -i <data_path>\n\n" \
    "Allowed options"

int main(int argc, char **argv) {
    // Common, everyday options.
    po::options_description general("General options");
    general.add_options()                    //
            ("help,h", "produce help message")  //
            ("input,i", po::value<string>(), "Input directory with env.rsc and robot.rsc files")
            ("device,d", po::value<string>()->default_value("cpu"), "Device: 'cpu' or 'cuda:N'")
            ("max_time,m", po::value<double>()->default_value(1000000), "Max simulation time (seconds)")
            ("record_size,r", po::value<int>()->default_value(5), "Record buffer size")
            ("save_record,S", po::value<int>()->default_value(1), "Save records (0|1)")
            ("log_level,l", po::value<string>()->default_value("info"),
             "Log level: err, warn, info, debug, trace");

    // Advanced solver-tuning knobs.
    po::options_description tuning("Solver tuning options (advanced)");
    tuning.add_options()  //
        ("compute_accuracy",
         po::value<string>()->default_value("float32"),
         "Simulation compute accuracy (float32 | float64)")(
            "memory_accuracy",
            po::value<string>()->default_value("float32"),
            "Simulation memory accuracy (float32 | float64)"
        );

    po::options_description desc(APP_DESCRIPTION);
    desc.add(general).add(tuning);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") || !vm.count("input")) {
        cout << desc << "\n";
        return 1;
    }

    fs::path input(vm["input"].as<string>());

    if (!fs::is_directory(input)) {
        cerr << "Error: input directory not found\n";
        return 1;
    }

    // Parse device
    RS_DeviceLocation device;
    string device_str = vm["device"].as<string>();
    if (device_str == "cpu") {
        device = {RSE_DEVICE_TYPE_CPU, 0};
    } else if (device_str.find("cuda:") == 0) {
        device.type = RSE_DEVICE_TYPE_CUDA;
        device.id = stoul(device_str.substr(5));
    } else {
        cerr << "Error: invalid device: " << device_str << "\n";
        return 1;
    }

    RiseOrigin rise(device);

    // Load rsc files
    vector<string> rsc_files;
    for (auto &file : fs::directory_iterator(input)) {
        auto extension = boost::algorithm::to_lower_copy(file.path().extension().string());
        if (extension == ".rsct" || extension == ".rscb") {
            rsc_files.push_back(file.path().string());
        }
    }

    if (rsc_files.empty()) {
        cerr << "Error: no .rsct or .rscb files found in input directory\n";
        return 1;
    }

    // Load config from files
    std::cout << "Loading world...";
    RS_Timer timer;
    rsc::RSC_World world;
    world.assembleFrom(rsc_files);
    std::cout << "loaded in: " << timer.elapsedTimeString() << std::endl;

    // CLI flags override the world's simulation settings.
    auto &runtime = world.simulation->runtime;
    runtime.record_buffer_size = vm["record_size"].as<int>();
    runtime.max_time = vm["max_time"].as<double>();
    runtime.is_save_record = static_cast<bool>(vm["save_record"].as<int>());
    runtime.compute_accuracy = vm["compute_accuracy"].as<string>() == "float64" ? RSE_FLOAT_64
                                                                                : RSE_FLOAT_32;
    runtime.memory_accuracy = vm["memory_accuracy"].as<string>() == "float64" ? RSE_FLOAT_64
                                                                              : RSE_FLOAT_32;
    runtime.log_level = vm["log_level"].as<string>();

    // Start the world
    auto handle = rise.hello(
        world, {}  // callback (not exposable via CLI)
    );

    handle->waitForEnd();
    auto status = handle->getStatus();
    cout << "World ended with status: " << enumToString(status) << "\n";
    return status == RSE_WORLD_END_SUCCESS ? 0 : 1;
}
