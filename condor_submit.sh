cat <<EOF> condor_submit_desc
Executable   = /bin/bash
Universe     = vanilla
Environment = "PATH=/usr/bin:/bin"
EOF
if [ "$1" -eq 'mval2.mval' ]; then
    cat <<EOF>> condor_submit_desc
Requirements = (machine == \"mval2.mval\")
Request_cpus = 4
Request_gpus = 1
Request_memory = 20*1024
EOF
