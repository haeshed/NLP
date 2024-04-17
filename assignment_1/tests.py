# Create tests
def test_preprocess():
    return {
        'vocab_length': len(preprocess()),
    }

def test_lm():
    return {
        'english_2_gram_length': len(lm('en', 2, True)),
        'english_3_gram_length': len(lm('en', 3, True)),
        'french_3_gram_length': len(lm('fr', 3, True)),
        'spanish_3_gram_length': len(lm('es', 3, True)),
    }

def test_eval():
    return {
        'en_en': eval(lm('en', 3, True), 'en', 3),
        'en_fr': eval(lm('en', 3, True), 'fr', 3),
        'en_tl': eval(lm('en', 3, True), 'tl', 3),
        'en_nl': eval(lm('en', 3, True), 'nl', 3),
    }

def test_match():
    df = match()
    return {
        'en_en_3': df[(df['source'] == 'en') & (df['target'] == 'en') & (df['n'] == 3)]['perplexity'].values[0],
        'en_tl_3': df[(df['source'] == 'en') & (df['target'] == 'tl') & (df['n'] == 3)]['perplexity'].values[0],
        'en_nl_3': df[(df['source'] == 'en') & (df['target'] == 'nl') & (df['n'] == 3)]['perplexity'].values[0],
    }

def test_generate():
    return {
        'english_1_gram': generate('en', 1, "I", 20, 5), 
        'english_2_gram': generate('en', 2, "I am", 20, 5),
        'english_3_gram': generate('en', 3, "I am", 20, 5),
        'english_4_gram': generate('en', 4, "I Love", 20, 5),
        'spanish_2_gram': generate('es', 2, "Soy", 20, 5),
        'spanish_3_gram': generate('es', 3, "Soy", 20, 5),
        'french_2_gram': generate('fr', 2, "Je suis", 20, 5),
        'french_3_gram': generate('fr', 3, "Je suis", 20, 5),
    }

TESTS = [test_preprocess, test_lm, test_eval, test_match, test_generate]

# Run tests and save results
res = {}
for test in TESTS:
    try:
        cur_res = test()
        res.update({test.__name__: cur_res})
    except Exception as e:
        res.update({test.__name__: repr(e)})

with open('results.json', 'w') as f:
    json.dump(res, f, indent=2)

# Download the results.json file
files.download('results.json')