from googleapiclient import discovery
import pandas as pd
import time

API_KEY = 'AIzaSyABNdlXZ6-gOomLMUE6FJ1piSxG5DgnRqg'

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

# read test data
test_labeled = pd.read_csv("../Data/test_labeled.csv")
comment_text = test_labeled['comment_text']

# capture the error messages
errors = []

# helper function to get the toxicity score
def get_score(text):
    analyze_request = {
    'comment': { 'text': text },
    'requestedAttributes': {'TOXICITY': {}, 'SEVERE_TOXICITY':{}, 'PROFANITY':{}, 'THREAT':{}, 'INSULT':{}, 'IDENTITY_ATTACK':{}}
    }
    # some texts may trigger errors
    try:
        response = client.comments().analyze(body=analyze_request).execute()
        return list(map(lambda x: x['summaryScore']['value'], response['attributeScores'].values()))
    except Exception as e:
        errors.append(e)
        return [0]*6

# get toxicity scores
scores = []
# quota limit: 60 requests per minute
for i in range(0, len(comment_text), 60):
    # get start time
    start_time = time.time()
    scores += list(map(get_score, comment_text[i:min(i+60, len(comment_text))]))
    # sleep to not exceed the quota
    time.sleep(max(0, 60-time.time()+start_time))

    # store the results from time to time
    if i%1000==0:
        print(f'{i} entries have been processed')
        # store the results
        scores_df = pd.DataFrame(scores, columns = test_labeled.columns[2:])
        scores_df.to_csv('../prediction/Perspective/scores_perspective.csv', index=False)

# store the final results
scores_df = pd.DataFrame(scores, columns = test_labeled.columns[2:])
scores_df.to_csv('../prediction/Perspective/scores_perspective.csv', index=False)

# store the error messages
errors_df = pd.DataFrame(errors)
errors_df.to_csv('../prediction/Perspective/errors.csv', index=False)