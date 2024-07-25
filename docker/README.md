To build the container use the following command:
```
$ docker buildx build docker -f docker/Dockerfile.local -t mlir.local
```

Or this may take a form of
```
docker build docker --build-arg "http_proxy=$http_proxy" --build-arg "https_proxy=$https_proxy" --build-arg "no_proxy=.habana-labs.com" -f docker/Dockerfile.local -t mlir.local

```

To run the container use the following command:
```
$ docker run -ti mlir.local
```

To launch the test with the simulator use:
```
(cd ${WORK_ROOT}/fs/scripts && source setup.sh && umd_driver_env_variables_export && run_coral_fs -r -m umd) &> fs-sim.log &
source /opt/intel/oneapi/setvars.sh
ForceBCSForInternalCopyEngine=1 ForceBcsEngineIndex=1 ForceDeviceId=0B73 HardwareInfoOverride=1x4x8 IGC_Disable512GRFISA=0 NEOReadDebugKeys=1 OverrideSlmSize=320 PrintDebugSettings=1 ProductFamilyOverride=fcs RebuildPrecompiledKernels=1 KNOB_MAX_CORES_PER_NUMA_NODE=3 KNOB_XE3P_ENABLE_IGA_XE3PX=1 SetCommandStreamReceiver=2 TbxPort=1234 ./simulator_check
```
