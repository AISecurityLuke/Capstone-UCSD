#!/usr/bin/env bash
# =============================================================================
# replace_phrases_yellow.sh
# -----------------------------------------------------------------------------
# Replaces highly repetitive words/phrases in yellow_merged.json with a diverse
# set of 7 synonyms (mix of English & Spanish). Distributes each synonym evenly
# across all occurrences.
# =============================================================================

set -euo pipefail

INPUT=${1:-yellow_merged.json}
OUTPUT=${2:-yellow_merged_augmented.json}

export INPUT_FILE_PATH="$INPUT"
export OUTPUT_FILE_PATH="$OUTPUT"

python - << 'PY'
import json, random, re, os, collections

input_file=os.environ['INPUT_FILE_PATH']
output_file=os.environ['OUTPUT_FILE_PATH']

# Replacements: each original → 7 variants
replacements={
    "about": ["regarding","concerning","sobre","acerca de","with respect to","relativo a","en cuanto a"],
    "situation": ["scenario","circumstance","situación","contexto","estado","condición","coyuntura"],
    "me": ["myself","yo","a mí","para mí","moi","a mi persona","yours truly"],
    "how": ["cómo","de qué manera","in what way","de qué forma","qué tan","the way","how exactly"],
    "this situation": ["esta situación","este escenario","esta circunstancia","este caso","este contexto","esta coyuntura","esta condición"],
    "might": ["could","podría","may","tal vez","quizá","puede que","quizás"],
    "how do": ["cómo hago","cómo puedo","cómo se","de qué manera hago","cómo haré","cómo debes","cómo haces"],
    "moral": ["ethical","ético","righteous","morality-based","valores","rectitud","principled"],
    "right": ["correct","correcto","proper","adecuado","cierto","justo","exact"],
    "discovered": ["found","uncovered","descubierto","hallado","detected","unearthed","identificado"],
    "ethical": ["moral","ético","justo","responsible","deontological","upright","integro"],
    "involved": ["implicado","involucrado","engaged","included","concerned","participante","implicated"],
    "discovered my": ["found my","uncovered my","descubrió mi","halló mi","detectó mi","descubrí mi","identifiqué mi"],
    "im": ["I am","soy","estoy","yo soy","yo estoy","I'm","I am currently"],
    "really": ["truly","realmente","de verdad","indeed","actually","en realidad","genuinely"],
    "because discovered": ["since found","because found","porque descubrió","debido a que descubrió","as we discovered","ya que descubrimos","puesto que encontramos"],
    "make": ["create","produce","hacer","fabricar","generar","elaborar","construir"],
    "people": ["personas","gente","folks","individuals","humans","ciudadanos","persons"],
    "because discovered my": ["since I found my","because I found my","porque descubrí mi","debido a que encontré mi","ya que hallé mi","puesto que detecté mi","as I discovered my"],
    "potential": ["potencial","capability","ability","capacity","promise","aptitude","possibility"],
}

# Read data
with open(input_file,'r',encoding='utf-8') as fh:
    data=json.load(fh)

# Build replacement queues
queues={}
# Sort originals by length descending to handle longer phrases first
for original in sorted(replacements.keys(), key=len, reverse=True):
    variants=replacements[original]
    # regex pattern
    if ' ' in original:
        pattern=re.compile(re.escape(original),flags=re.IGNORECASE)
    else:
        pattern=re.compile(r'\b'+re.escape(original)+r'\b',flags=re.IGNORECASE)
    total=0
    for obj in data:
        total+=len(pattern.findall(obj.get('user_message','')))
    base=total//len(variants)
    extra=total%len(variants)
    repl_list=[v for i,v in enumerate(variants) for _ in range(base + (1 if i<extra else 0))]
    random.shuffle(repl_list)
    queues[original]=collections.deque(repl_list)

# Replacement pass
for obj in data:
    text=obj.get('user_message','')
    for original in sorted(replacements.keys(), key=len, reverse=True):
        queue=queues[original]
        if not queue:
            continue
        if ' ' in original:
            pat=re.compile(re.escape(original),flags=re.IGNORECASE)
        else:
            pat=re.compile(r'\b'+re.escape(original)+r'\b',flags=re.IGNORECASE)
        def repl(match, q=queue):
            return q.popleft() if q else match.group(0)
        text=pat.sub(repl,text)
    obj['user_message']=text

with open(output_file,'w',encoding='utf-8') as fh:
    json.dump(data,fh,ensure_ascii=False,indent=2)

# print finished
print(f'Done. Output -> {output_file}')
PY