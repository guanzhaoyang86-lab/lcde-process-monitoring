function metadata = generate_pytep_run(output_path, simulator_dir, seed_value, fault_id, magnitude, duration_hours)
%GENERATE_PYTEP_RUN Generate one governed pyTEP trajectory without editing the model.
%
% The function uses the frozen SoftwareX pyTEP simulator in simulator_dir,
% keeps the published model structure flag (227), and writes a compact MAT
% file containing the canonical [time, 41 XMEAS, 12 XMV] process matrix.

arguments
    output_path (1, 1) string
    simulator_dir (1, 1) string
    seed_value (1, 1) double {mustBeFinite, mustBePositive, mustBeInteger}
    fault_id (1, 1) double {mustBeFinite, mustBeInteger, mustBeGreaterThanOrEqual(fault_id, 0), mustBeLessThanOrEqual(fault_id, 28)}
    magnitude (1, 1) double {mustBeFinite, mustBeGreaterThanOrEqual(magnitude, 0), mustBeLessThanOrEqual(magnitude, 1)}
    duration_hours (1, 1) double {mustBeFinite, mustBePositive}
end

if fault_id == 0 && magnitude ~= 0
    error("generate_pytep_run:HealthyMagnitude", ...
        "A healthy trajectory (fault_id=0) must use magnitude=0.");
end
if fault_id > 0 && magnitude <= 0
    error("generate_pytep_run:FaultMagnitude", ...
        "A faulty trajectory must use a strictly positive magnitude.");
end
if seed_value > flintmax
    error("generate_pytep_run:SeedPrecision", ...
        "seed_value must be exactly representable as a MATLAB double.");
end

simulator_dir = string(java.io.File(char(simulator_dir)).getCanonicalPath());
output_path = string(java.io.File(char(output_path)).getCanonicalPath());
if ~isfolder(simulator_dir)
    error("generate_pytep_run:MissingSimulator", ...
        "Simulator directory does not exist: %s", simulator_dir);
end
output_dir = fileparts(output_path);
if ~isfolder(output_dir)
    mkdir(output_dir);
end

original_dir = string(pwd);
directory_cleanup = onCleanup(@() cd(original_dir));
cd(simulator_dir);
addpath(genpath(simulator_dir));

cache_root = fullfile(output_dir, "matlab_cache");
if ~isfolder(cache_root)
    mkdir(cache_root);
end
Simulink.fileGenControl("set", ...
    "CacheFolder", cache_root, ...
    "CodeGenFolder", cache_root, ...
    "createDir", true);

load_system("TElib");
load_system("MultiLoop_mode3");
model_cleanup = onCleanup(@() close_models_without_saving());

% Simulink resolves the published model inputs from MATLAB's base workspace,
% even though this governed wrapper is itself a function.
evalin("base", "loadSimEnvironment;");
seed = double(seed_value);
Ts_save = 0.05; % three-minute sampling
assignin("base", "seed", seed);
assignin("base", "Ts_save", Ts_save);

idv_init = zeros(3, 28);
if fault_id > 0
    idv_init(2, fault_id) = magnitude;
    idv_init(3, fault_id) = 30.0;
end
assignin("base", "idv_init", idv_init);
evalin("base", "init_setpointinput_from_workspace;");
evalin("base", "init_idvinput_from_workspace;");

plant_block = "MultiLoop_mode3/TE Plant/TE Code";
if string(get_param(plant_block, "FunctionName")) ~= "temexd_mod"
    error("generate_pytep_run:UnexpectedPlant", ...
        "The TE plant S-function is not temexd_mod.");
end
if string(get_param(plant_block, "Parameters")) ~= ...
        "xInitial.signals(1).values,seed,227"
    error("generate_pytep_run:StructureFlag", ...
        "The frozen pyTEP model no longer uses structure flag 227.");
end

set_param("MultiLoop_mode3", "StopTime", sprintf("%.17g", duration_hours));
simulation_output = sim("MultiLoop_mode3", "ReturnWorkspaceOutputs", "on");

tout = double(simulation_output.tout(:));
simout = double(simulation_output.simout);
xmv = double(simulation_output.xmv);
if size(simout, 1) ~= numel(tout) || size(simout, 2) ~= 41
    error("generate_pytep_run:ProcessShape", ...
        "Expected simout shape N-by-41.");
end
if size(xmv, 1) ~= numel(tout) || size(xmv, 2) ~= 12
    error("generate_pytep_run:ManipulatedShape", ...
        "Expected xmv shape N-by-12.");
end
processdata = [tout, simout, xmv]; %#ok<NASGU>
if any(~isfinite(processdata), "all")
    error("generate_pytep_run:NonFinite", ...
        "Generated processdata contains non-finite values.");
end

model_version = string(get_param("MultiLoop_mode3", "ModelVersion")); %#ok<NASGU>
matlab_release = string(version("-release")); %#ok<NASGU>
simulink_version = string(ver("Simulink").Version); %#ok<NASGU>
sample_period_hours = 0.05; %#ok<NASGU>
fault_onset_hours = 30.0; %#ok<NASGU>
structure_flag = 227; %#ok<NASGU>
source_simulator_dir = simulator_dir; %#ok<NASGU>
generated_utc = string(datetime("now", "TimeZone", "UTC", ...
    "Format", "yyyy-MM-dd'T'HH:mm:ss.SSSXXX")); %#ok<NASGU>

save(output_path, "processdata", "idv_init", "seed", "fault_id", ...
    "magnitude", "duration_hours", "sample_period_hours", ...
    "fault_onset_hours", "structure_flag", "model_version", ...
    "matlab_release", "simulink_version", "source_simulator_dir", ...
    "generated_utc", "-v7");

metadata = struct( ...
    "output_path", output_path, ...
    "rows", size(processdata, 1), ...
    "columns", size(processdata, 2), ...
    "final_time_hours", tout(end), ...
    "seed", seed, ...
    "fault_id", fault_id, ...
    "magnitude", magnitude, ...
    "structure_flag", structure_flag, ...
    "model_version", model_version, ...
    "matlab_release", matlab_release, ...
    "simulink_version", simulink_version);

clear model_cleanup directory_cleanup;
end

function close_models_without_saving()
if bdIsLoaded("MultiLoop_mode3")
    close_system("MultiLoop_mode3", 0);
end
if bdIsLoaded("TElib")
    close_system("TElib", 0);
end
end
