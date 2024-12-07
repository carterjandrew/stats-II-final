{ pkgs ? import <nixpkgs> {}}:
let
  fhs = pkgs.buildFHSUserEnv {
    name = "my-fhs-environment";

    targetPkgs = _: [
      pkgs.micromamba
    ];

    profile = ''
      set -e
      eval "$(micromamba shell hook --shell=posix)"
      export MAMBA_ROOT_PREFIX=${builtins.getEnv "PWD"}/.mamba
      micromamba create -y -q -n stats-env
      micromamba activate stats-env
      micromamba install pytorch --yes -f requirements.txt -c pytorch -c conda-forge
      set +e
    '';
  };
in fhs.env
