#!/usr/bin/env bash

# sets up a local environment based on CMSSW that provides all required software

action() {
    # CMSSW variables
    export SRAM_ARCH="slc7_amd64_gcc820"
    export CMSSW_VERSION="CMSSW_11_1_0_pre4"

    # temporary variables
    local origin="$PWD"
    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"
    local python_version="$( python -c "import sys; print('{0.major}.{0.minor}'.format(sys.version_info))" )"
    local software_base="$this_dir/.tmp"
    local cmssw_base="$software_base/$CMSSW_VERSION"

    # setup CMSSW
    source /cvmfs/cms.cern.ch/cmsset_default.sh ""
    if [ ! -d "$cmssw_base/src" ]; then
        echo "setting up CMSSW environment in $cmssw_base"
        mkdir -p "$( dirname "$cmssw_base" )"
        cd "$( dirname "$cmssw_base" )"
        scramv1 project CMSSW "$CMSSW_VERSION"
        cd "$CMSSW_VERSION/src"
    else
        cd "$cmssw_base/src"
    fi
    eval `scramv1 runtime -sh`
    cd "$origin"

    # minimal software stack
    if [ ! -d "$software_base/lib" ]; then
        echo "setting up minimal software stack in $software_base"
        pip install --ignore-installed --no-cache-dir --prefix "$software_base" oyaml
        pip install --ignore-installed --no-cache-dir --prefix "$software_base" tabulate
    fi

    # update PATH and PYTHONPATH
    export PATH="$this_dir/bin:$software_base/bin:$PATH"
    export PYTHONPATH="$this_dir:$software_base/lib/python${python_version}/site-packages:$PYTHONPATH"
}
action "$@"
