import subprocess
import os

def run_batch_processor_dry(exe_path, filename, args, dryrun):
  '''
  Runs the batch processor executable with which runs the audio
  processing and creates audio files of all debug streams.
  '''
  if not args:
    args = ['1']
  if '.wav' in filename:
    raise Exception('Filenames should be given without file extensions')
  # find executable based on our system
  if not os.path.isfile(exe_path):
    raise Exception(f'Executable not found: "{exe_path}"')
  args = [exe_path, 'samples/processed/', filename] + args 
  print(' '.join(args))
  if dryrun:
    return
  output = subprocess.run(args, capture_output=True)
  if output.returncode != 0:
    print(' '.join(output.args))
    raise Exception(output.stderr)
  
path = 'exe/'
file = 'sounds/clean'
run_batch_processor_dry(path, file, None, False)