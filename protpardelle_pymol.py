"""
Protpardelle Pymol Extension 

load protpardelle_pymol
(protpardelle_setlink link ) (if you want to run a local server)

Conditional:
fetch 1pga
select some residues
generate 5 samples resampling the residues in sele
protpardelle 1pga, sele, 5  

unconditional: 
Sample sequences between length 50 and 60 with step size 5 and generate 1 sample per sampled length. Use the allatom model
protpardelle_uncond 50, 60, 5, 1, allatom
"""

from pymol import cmd

import os
import json
import time
import threading

try:
    from gradio_client import Client
except ImportError:
    print("gradio_client not installed, trying install:")
    import pip

    pip.main(["install", "gradio_client"])
    from gradio_client import Client


if os.environ.get("GRADIO_LOCAL") != None:
    public_link = "http://127.0.0.1:7860"
else:
    public_link = "ProteinDesignLab/protpardelle"


def thread_protpardelle(
    input_pdb,
    resample_idxs,
    modeltype,
    mode,
    minlen=50,
    maxlen=60,
    steplen=2,
    per_len=2,
):
    client = Client(public_link)

    job = client.submit(
        input_pdb,  # str in 'PDB Content' Textbox component
        modeltype,  # str in 'Choose a Mode' Radio component
        f'"{resample_idxs}"',  # str in 'Resampled Idxs' Textbox component
        mode,  # str (Option from: ['backbone', 'allatom'])
        minlen,  # int | float (numeric value between 2 and 200) minlen
        maxlen,  # int | float (numeric value between 3 and 200) in 'maxlen' Slider component
        steplen,  # int | float (numeric value between 1 and 50) in 'steplen' Slider component
        per_len,  # int | float (numeric value between 1 and 200) in 'perlen' Slider component
        api_name="/protpardelle",
    )
    # start time
    start = time.time()

    while job.done() == False:
        status = job.status()
        elapsed = time.time() - start
        # format as hh:mm:ss
        elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        print(f"\r protpardelle running since {elapsed}", end="")
        time.sleep(1)
    results = job.result()

    # load each result into pymol
    results = json.loads(results)

    for name, pdb_content in results:
        print(name)
        cmd.read_pdbstr(pdb_content, os.path.basename(name))


def query_protpardelle(
    name_of_input: str,
    selection_resample_idxs: str = "",
    per_len: int = 2,
    mode: str = "allatom",
):
    """
    AUTHOR
    Simon Duerr
    https://twitter.com/simonduerr
    DESCRIPTION
    Run Protpardelle
    USAGE
    protpardelle name_of_input, selection_resampled_idx, modeltype, mode, per_len
    PARAMETERS
    name_of_input = string: name of input object
    selection_resampled_idx = string: selection of resampled protein residues
    per_len = int: per_len (default: 2)
    mode = string: mode (default: 'allatom')
    """
    if name_of_input != "":
        input_pdb = cmd.get_pdbstr(name_of_input)

        all_aa = cmd.index(name_of_input + " and name CA")
        idx = cmd.index(selection_resample_idxs + " and name CA")

        # map to zero indexed values
        aa_mapping = {aa[1]: i for i, aa in enumerate(all_aa)}

        idx = ",".join([str(aa_mapping[aa[1]]) for aa in idx])

        print("resampling", idx, "(zero indexed) from", name_of_input)

    t = threading.Thread(
        target=thread_protpardelle,
        args=(input_pdb, idx, "conditional", mode),
        kwargs={"per_len": per_len},
        daemon=True,
    )
    t.start()


def query_protpardelle_uncond(
    minlen: int = 50,
    maxlen: int = 60,
    steplen: int = 2,
    per_len: int = 2,
    mode: str = "allatom",
):
    """
    AUTHOR
    Simon Duerr
    https://twitter.com/simonduerr
    DESCRIPTION
    Run Protpardelle
    USAGE
    protpardelle_uncond minlen, maxlen, steplen, per_len,mode
    PARAMETERS
    minlen = int: minlen
    maxlen = int: maxlen
    steplen = int: steplen
    per_len = int: per_len
    mode = string: mode (default: 'allatom')
    """

    modeltype = "unconditional"
    idx = None
    input_pdb = None

    t = threading.Thread(
        target=thread_protpardelle,
        args=(input_pdb, idx, modeltype, mode),
        kwargs={
            "minlen": minlen,
            "maxlen": maxlen,
            "steplen": steplen,
            "per_len": per_len,
        },
        daemon=True,
    )
    t.start()


def setprotpardellelink(link: str):
    """
    AUTHOR
    Simon Duerr
    https://twitter.com/simonduerr
    DESCRIPTION
    Set a public link to use a locally hosted version of this space
    USAGE
    protpardelle_setlink link_or_username/spacename
    """

    global public_link
    try:
        client = Client(link)
    except:
        print("could not connect to:", public_link)

    public_link = link


cmd.extend("protpardelle_setlink", setprotpardellelink)

cmd.extend("protpardelle", query_protpardelle)

cmd.extend("protpardelle_uncond", query_protpardelle_uncond)
