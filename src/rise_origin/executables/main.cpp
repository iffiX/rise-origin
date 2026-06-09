// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <fmt/format.h>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <iostream>

#include "rise_origin/rise_origin.h"
#include "rise_origin/utils/rs_timer.h"

namespace fs = boost::filesystem;
namespace po = boost::program_options;
using namespace std;
using namespace fmt;
using namespace rise;

#define APP_DESCRIPTION                     \
    "RiseOrigin Simulator\n\n"             \
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
            ("dt_update_interval", po::value<unsigned int>()->default_value(10),
             "Time-step update interval")
            ("rigidbody_update_interval", po::value<unsigned int>()->default_value(10),
             "Rigid body update interval")
            ("contact_update_interval", po::value<unsigned int>()->default_value(4),
             "Contact detection update interval")
            ("mechanical_signal_update_interval", po::value<unsigned int>()->default_value(10),
             "Mechanical signal update interval")
            ("divergence_check_interval", po::value<unsigned int>()->default_value(100),
             "Divergence check interval")
            ("compute_accuracy", po::value<string>()->default_value("float32"),
             "Simulation compute accuracy (float32 | float64)")
            ("memory_accuracy", po::value<string>()->default_value("float32"),
             "Simulation memory accuracy (float32 | float64)");

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
    if (!fs::is_regular_file(input / "env.rsc")) {
        cerr << "Error: env.rsc not found in input directory\n";
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

    // Load first robot config
    string env_config = (input / "env.rsc").string();
    string robot_config;

    for (auto &file : fs::directory_iterator(input)) {
        if (boost::algorithm::to_lower_copy(file.path().extension().string()) == ".rsc" &&
            boost::algorithm::to_lower_copy(file.path().filename().string()) != "env.rsc") {
            robot_config = file.path().string();
            break;
        }
    }

    if (robot_config.empty()) {
        cerr << "Error: no robot.rsc file found in input directory\n";
        return 1;
    }

    // Load config from files
    std::cout << "Loading config...";
    RS_Timer timer;
    RS_Config config;
    config.open({env_config, robot_config}, true);
    std::cout << "loaded in: " << timer.elapsedTimeString() << std::endl;

    // CLI flags override the config's runtime_* execution settings.
    auto &sim = config.simulation_config;
    sim.runtime_dt_update_interval = vm["dt_update_interval"].as<unsigned int>();
    sim.runtime_rigidbody_update_interval = vm["rigidbody_update_interval"].as<unsigned int>();
    sim.runtime_contact_update_interval = vm["contact_update_interval"].as<unsigned int>();
    sim.runtime_mechanical_signal_update_interval =
        vm["mechanical_signal_update_interval"].as<unsigned int>();
    sim.runtime_divergence_check_interval = vm["divergence_check_interval"].as<unsigned int>();
    sim.runtime_record_buffer_size = vm["record_size"].as<int>();
    sim.runtime_max_time = vm["max_time"].as<double>();
    sim.runtime_save_record = static_cast<bool>(vm["save_record"].as<int>());
    sim.runtime_simulation_compute_accuracy =
        vm["compute_accuracy"].as<string>() == "float64" ? RSE_FLOAT_64 : RSE_FLOAT_32;
    sim.runtime_simulation_memory_accuracy =
        vm["memory_accuracy"].as<string>() == "float64" ? RSE_FLOAT_64 : RSE_FLOAT_32;
    sim.runtime_log_level = vm["log_level"].as<string>();

    // Run simulation
    auto handle = rise.runSim(
        config,
        0,  // kernel id (single-device, always 0)
        {}  // callback (not exposable via CLI)
    );

    handle->waitForEnd();
    auto status = handle->getStatus();
    cout << "Simulation completed with status: " << enumToString(status) << "\n";
    return status == RSE_SIMULATION_END_SUCCESS ? 0 : 1;
}
