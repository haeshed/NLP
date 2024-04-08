# Create tests
def test_preprocess():
    return {
        'vocab_length': len(preprocess()),
    }

def test_lm():
    return {
        'english_2_gram_length': len(lm('en', 2)),
        'english_3_gram_length': len(lm('en', 3)),
        'french_3_gram_length': len(lm('fr', 3)),
        'spanish_3_gram_length': len(lm('es', 3)),
    }

def test_eval():
    return {
        'english_on_english': round(eval(lm('en', 3), 'en'), 2),
        'english_on_french': round(eval(lm('en', 3), 'fr'), 2),
        'english_on_spanish': round(eval(lm('en', 3), 'es'), 2),
    }

def test_match():
    df = match()
    return {
        'df_shape': df.shape,
        'en_en_1': df[(df['source'] == 'en') & (df['target'] == 'en') & (df['n'] == 1)]['perplexity'].values[0],
        'tl_tl_1': df[(df['source'] == 'tl') & (df['target'] == 'tl') & (df['n'] == 1)]['perplexity'].values[0],
        'tl_nl_4': df[(df['source'] == 'tl') & (df['target'] == 'nl') & (df['n'] == 4)]['perplexity'].values[0],
    }

def test_generate():
    return {
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