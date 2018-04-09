import os
import re
import json
import urllib.request
import numpy as np
import pandas as pd
from EmotionDetection import EmotionDetection


# function to delete url
def del_url(line):
    return re.sub(r'(\S*(\.com).*)|(https?:\/\/.*)', "", line)


# replace tag
def check_tag(line):
    return re.sub(r'\@|\#', "", line)


# Some special character
def check_special(line):
    return line.replace('♡', 'love ').replace('\"', '').replace('“', '').replace('”', '').replace('…', '...')


def sendto140(line):
    url = 'http://www.sentiment140.com/api/bulkClassifyJson'
    query_dict = {"data": [{"text": check_special(check_tag(del_url(str(line).strip())))}]}
    params = json.dumps(query_dict).encode('utf8')
    try:
        req = urllib.request.Request(url, params)
        res = urllib.request.urlopen(req, timeout=5)

        query_result = json.loads(res.read().decode('utf8'))
        return sentiment_dict[int(query_result["data"][0]["polarity"])]
    except:
        print("Fail : " + line)
        return sentiment_dict[2]


sentiment_dict = {
    0: -1,
    2: 0,
    4: 1
}

ed = EmotionDetection()

if __name__ == "__main__":

    dir_path = '../datasets/depression_raw_pickle/'
    out_dir_path = '../datasets/depression_senti_emo_pickle/'
    files = os.listdir(dir_path)
    done_files = os.listdir(out_dir_path)
    print('Reading Patients, total:{}'.format(len(files)))

    # Read all timelines from pickles
    depression_timelines = [pd.read_pickle(dir_path + file) for file in files]

    for i, timeline in enumerate(depression_timelines):

        if timeline.uid[0] in done_files: continue # check repeat

        total = timeline.shape[0]
        senti_array = np.zeros(total)
        emo_array = [[],[],[]]

        # go through each tweet
        for j, tweet in enumerate(timeline.tweet):
            # sentiments query
            senti_res = sendto140(tweet)
            senti_array[j] = senti_res

            # emotions query
            emotion_res = ed.get_emotion_json(tweet)
            emotion1, ambiguous = emotion_res[u'groups'][0][u'name'], emotion_res[u'ambiguous']

            if len(emotion_res[u'groups']) == 2:
                emotion2 = emotion_res[u'groups'][1][u'name']
            else:
                emotion2 = emotion1

            emo_array[0].append(emotion1)
            emo_array[1].append(emotion2)
            emo_array[2].append(ambiguous)

            print("{}({}/{}) - sentiment:{:2d}, emotion:{:4s}/{:4s}/{:3s}, tweet:{}".format(i, j + 1, total, senti_res, emotion1[:4], emotion2[:4], ambiguous, tweet))

        # append result to timeline dataframe
        timeline['senti'] = senti_array
        timeline['emotion1'] = emo_array[0]
        timeline['emotion2'] = emo_array[1]
        timeline['ambiguous'] = emo_array[2]

        # Save file
        timeline.to_pickle(out_dir_path + timeline.uid[0])


