import sys, os
import time
import re
import argparse
import pickle

##############################################################################################################
#
# MPI - Cluster Job Submission Script
#
##############################################################################################################

sub_fmt = '''executable = /bin/sh
args = {0}
getenv = True
error = {1}
output = {2}
log = {3}
queue {4}'''


GPU_NAMES = {
    '2080': 'GeForce RTX 2080 Ti',
    '2080ti': 'GeForce RTX 2080 Ti',
    '2080Ti': 'GeForce RTX 2080 Ti',

    'K80': 'Tesla K80',

    'K20': 'Tesla K20Xm',

    'V100-16': 'Tesla V100-PCIE-16GB',

    'V100-32-pci': 'Tesla V100-PCIE-32GB',
    'V100-32-PCI': 'Tesla V100-PCIE-32GB',
    'V100-32-p': 'Tesla V100-PCIE-32GB',

    'V100-32-s': 'Tesla V100-PCIE-32GB',
    'V100-32-sxm': 'Tesla V100-PCIE-32GB',
    'V100-32-sxm2': 'Tesla V100-PCIE-32GB',
    'V100-32-SXM': 'Tesla V100-PCIE-32GB',
    'V100-32-SXM2': 'Tesla V100-PCIE-32GB',
    'V100-32': 'Tesla V100-SXM2-32GB',

    'P40': 'Tesla P40',
    'P100': 'Tesla P100-PCIE-16GB',
}

job_template = '''
#!/bin/sh

export RESTART_AFTER="10"
echo "-- starting job --"

{}

if [ $? -eq 3 ]
then
  echo "-- pausing for restart --"
  exit 3
fi

echo "-- job complete --"
'''


def write_job(cmds, path, name=None, cddir=None, tmpl=None):
    with open(path, 'w') as f:
        if name is not None:
            f.write('\n# Job script for {}\n\n'.format(name))

        if cddir is not None:
            f.write('cd {}\n'.format(cddir))
        if tmpl is None:
            f.writelines(cmds)
        else:
            f.write(tmpl.format('\n'.join(cmds)))
        f.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Create a submission script for the cluster')
    parser.add_argument('--name', type=str, default=None,
                        help='Name of job')
    parser.add_argument('--no-date', dest='use_date', action='store_false',
                        help='Dont use date/time in name')

    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Make the job interactive.')

    parser.add_argument('--use-template', action='store_true',
                        help='Use local job template')
    parser.add_argument('--array', action='store_true',
                        help='Treat commands as an array')

    parser.add_argument('--queue', type=int, default=0,
                        help='Queue number for the job arrays')

    parser.add_argument('--no-output', dest='use_out', action='store_false',
                        help='Dont log stdout')

    parser.add_argument('--cpu', type=int, default=1,
                        help='number of cpus')
    parser.add_argument('--mem', type=int, default=32,
                        help='memory in GB')
    parser.add_argument('--gpu', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--faster', action='store_true')
    parser.add_argument('--gpu-names', type=str, nargs='+', default=None)

    parser.add_argument('--bid', type=int, default=50,
                        help='the bid (this automatically submits the job after prep)')

    parser.add_argument('--cmd', type=str, default=None,
                        help='executable command')
    parser.add_argument('--script', type=str, default=None,
                        help='path to sh script')
    parser.add_argument('--redo', type=str, default=None,
                        help='path to a job dir or sh file to rerun')

    parser.add_argument('--restart-after', type=float, default=None,
                        help='time in hours to wait before restarting job (to reset costs, good choices are 1-3)')

    parser.add_argument('--dir', type=str, default=None,
                        help='path to change to before executing job')
    parser.add_argument('--root', type=str, default='jobs',
                        help='path to jobs folder')

    args = parser.parse_args(argv)
    print(args.array, args.interactive)

    past_jobs = os.listdir(args.root)

    assert args.cmd is not None or args.script is not None or args.redo is not None, 'nothing to run'

    num = len(past_jobs)

    if args.name is None:
        args.name = 'job{}'.format(str(num).zfill(4))

    if args.use_date:
        now = time.strftime("%y%m%d-%H%M%S")
        args.name = '{}_{}'.format(args.name, now)

    path = os.path.join(args.root, args.name)

    try:
        os.makedirs(path)
    except FileExistsError:
        pass

    job_path = None
    if args.redo is not None:
        if os.path.isdir(args.redo):
            args.redo = os.path.join(args.redo, 'job.sh')
        job_path = args.redo
    else:
        cmds = []
        if args.script is not None:
            with open(args.script, 'r') as f:
                cmds.extend([l for l in f.readlines() if len(l) > 1 and l[0] != '#'])

        if args.cmd is not None:
            cmds.append(args.cmd + '\n')
        assert len(cmds)

        if args.array:
            print('Found {cm} commands, will submit {num} replicas'.format(cm=len(cmds), num=args.queue))
            for i in range(args.queue):
                jpath = os.path.join(path, 'job_{}.sh'.format(i))
                write_job([args.cmd + " -eid " + str(i) + '\n'], jpath, name=args.name + ' - process: {}'.format(i), cddir=args.dir, tmpl=job_template if args.use_template else None)

            job_path = os.path.join(path, 'job_$(Process).sh')

            args.array = len(cmds)
            if args.queue > 0:
                args.array = args.queue
        else:
            job_path = os.path.join(path, 'job.sh')

            write_job(cmds, job_path, name=args.name, cddir=args.dir, tmpl=job_template if args.use_template else None)

    sub = []

    sub.append(
        'environment = JOBDIR={};JOBEXEC={};PROCESS_ID=$(Process);JOB_ID=$(ID);JOB_NUM={}'.format(path, job_path, num))

    reqs = []

    assert args.mem > 0
    sub.append('request_memory = {}'.format(args.mem * 1024))
    sub.append('request_cpus =  {}'.format(args.cpu))
    if args.gpu > 0:
        sub.append('request_gpus =  {}'.format(args.gpu))
        if args.gpu_names is not None:
            reqs.append(' || '.join('CUDADeviceName == \"{}\"'.format(GPU_NAMES[gname]) for gname in args.gpu_names))
            print('Requiring: {}'.format(' or '.join(args.gpu_names)))
        if args.fast:
            print('Fast job')
            reqs.append('CUDAGlobalMemoryMb > 18000')
        if args.faster:
            print('Faster job')
            reqs.append('CUDAGlobalMemoryMb > 26000')

    if len(reqs):
        sub.append('requirements = {}'.format(' && '.join('({})'.format(r) for r in reqs)))

    if args.restart_after is not None:
        sub.append('''MaxTime = {}
                        periodic_hold = (JobStatus =?= 2) && ((CurrentTime - JobCurrentStartDate) >= $(MaxTime))
                        periodic_hold_reason = "Job runtime exceeded"
                        periodic_hold_subcode = 1'''.format(int(args.restart_after * 3600)))

    # restart when command exits with 3
    sub.append('''on_exit_hold = (ExitCode =?= 3)
                    on_exit_hold_reason = "Checkpointed, will resume"
                    on_exit_hold_subcode = 2
                    periodic_release = ( (JobStatus =?= 5) && (HoldReasonCode =?= 3) && ((HoldReasonSubCode =?= 1) || (HoldReasonSubCode =?= 2)) )''')

    stdoutname = 'stdout-$(Process).txt' if args.array else 'stdout.txt'
    logname = 'log-$(Process).txt' if args.array else 'log.txt'

    sub.append(sub_fmt.format(job_path, os.path.join(path, stdoutname),
                              os.path.join(path, stdoutname),
                              os.path.join(path, logname),
                              args.array if args.array is not False else ''))

    sub_path = os.path.join(path, 'submit.sub')
    with open(sub_path, 'w') as f:
        f.write('\n'.join(sub))

    pickle.dump(args, open(os.path.join(path, 'args.pkl'), 'wb'))

    print('Job {} prepared'.format(args.name))

    if args.bid is not None:
        os.system(
            'condor_submit_bid {bid} {job}{i}'.format(bid=args.bid, job=sub_path, i=' -i' if args.interactive else ''))

        print('Job submitted with a bid: {}'.format(args.bid))


if __name__ == '__main__':
    main()
