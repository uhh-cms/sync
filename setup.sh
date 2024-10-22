#!/usr/bin/env bash

# sets up a virtual env based on the system python (>=3.9) and configures a few variables
action() {
    local shell_is_zsh=$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    # zsh options
    if ${shell_is_zsh}; then
        emulate -L bash
        setopt globdots
    fi

    #
    # global variables
    # (most variables respect defaults exported previously!)
    #

    export SYNC_DIR="${this_dir}"
    export SYNC_DATA_DIR="${SYNC_DATA_DIR:-${this_dir}/data}"
    export SYNC_SOFTWARE_DIR="${SYNC_SOFTWARE_DIR:-${SYNC_DATA_DIR}/software}"

    export LANGUAGE="${LANGUAGE:-en_US.UTF-8}"
    export LANG="${LANG:-en_US.UTF-8}"
    export LC_ALL="${LC_ALL:-en_US.UTF-8}"
    export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
    export GLOBUS_THREAD_MODEL="${GLOBUS_THREAD_MODEL:-none}"

    export PATH="${SYNC_DIR}/bin:${SYNC_SOFTWARE_DIR}/bin:${PATH}"
    export PYTHONPATH="${SYNC_DIR}:${PYTHONPATH}"


    #
    # setup virtual environment
    #

    local sync_venv_dir="${SYNC_SOFTWARE_DIR}/venv/sync"
    local venv_exists="$( [ -d "${sync_venv_dir}" ] && echo "true" || echo "false" )"

    if ! ${venv_exists}; then
        # detect python
        local ret
        local python_cmd
        local python_version
        local allowed_python_versions="3.9 3.10 3.11 3.12"
        local cmd
        for cmd in python python3; do
            ! command -v "${cmd}" > /dev/null && continue
            python_version="$( eval "${cmd} -c \"import sys;print('{0.major}.{0.minor}'.format(sys.version_info))\"" )"
            [ $? != "0" ] && continue
            [ "${python_version}" \< "3.9" ] && continue
            [[ "${allowed_python_versions}" != *"${python_version}"* ]] && continue
            python_cmd="${cmd}"
            break
        done
        if [ -z "${python_cmd}" ]; then
            >&2 echo "no suitable python version found after checking 'python', 'python3'"
            return "1"
        fi

        # create the venv
        echo -e "creating venv at \x1b[0;49;35m${sync_venv_dir}\x1b[0m with python \x1b[0;49;36m${python_version}\x1b[0m"
        eval "${python_cmd} -m venv \"${sync_venv_dir}\"" || return "$?"
    fi

    # source the venv
    source "${sync_venv_dir}/bin/activate" "" || return "$?"

    if ! ${venv_exists}; then
        # initial setup
        pip install -U pip setuptools wheel || return "$?"
        pip install -r "${SYNC_DIR}/requirements.txt" || return "$?"
    fi


    #
    # finalize
    #

    echo -e "\x1b[0;49;32msync tools\x1b[0m setup"
    # TODO: print hints about first steps
}

# entry point
action "$@"
