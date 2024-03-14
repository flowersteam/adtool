# Add a new remote server
You must have a remote server to start a remote experiment.<br/>
To access your server from the software you have to configure it in the [configs/remote_experiments](../configs/remote_experiments) folder.<br/>
Here you have two sub folders.<br/>
The first one, `additional_files`, contains the files we will execute to start the experiment on the remote server.<br/>
The second one, `profiles`, is used to define how to connect and interact with the remote server. <br/><br/>
We find there:<br/>

    - The current config name (we will choose it in the GUI).
    - The name of your ssh configuration, defined on your personnal file `/home/user/.ssh/config` (or other path you have set on services/.env file at the `SSH_CONFIG_FOLDER` var).
    - The work folder on the remote server
    - A local folder to store downloaded files created by the remote experiment before they are put in the database 
    - The command to start the experiment ( it will execute the associated file in "additional_file"). This command must echo `"[RUN_ID_start]"id"[RUN_ID_stop]"` in the remote shell either directly in the command or at the end of the associated additional_file. The example below does it on the additional file. Here id is equal to every PID the experiment launches (one for each seed).
    - The command to cancel all processes started by the experiment to run it self on the remote server.
    - Finally a number of second (to check that the experiment has started).

example:
```
        profiles:
            name: 
                myConfigNameInTheSoftware
            ssh_configuration: 
                myPersonnalShhConfigName
            work_path:
                /path/to/workdirectory/on/the/remote/server
            local_tmp_path:
                /tmp/adtool/remote_experiment/
            execution:
                bash additional_files/docker.sh $NB_SEEDS $ARGS > $EXPE_ID
            cancellation:
                scancel $RUN_ID
            check_experiment_launched_every:
                60

        additional_files:
            #!/bin/bash
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/anaconda3/lib/;

            PIDS=""
            for ((i=0; i<$1; i++))
            do
            PID=$(/root/anaconda3/envs/autoDiscTool/bin/python3 /home/workdir/adtool/adtool/run.py --seed ${i} ${@:2} >> experiment_logs.log & echo $!)
            if [[ $PIDS != "" ]]
            then
                PIDS+=" "
            fi
            PIDS+="${PID}"
            done
            echo "[RUN_ID_start]$PIDS[RUN_ID_stop]"
```

**Please note that you must first download and install the `adtool` lib on your remote server (follow [these steps](../README.md#autodisc-lib)) to launch experiments on this server.**