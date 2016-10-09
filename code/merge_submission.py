import pandas as pd
sex_submission = pd.read_csv('./data/gender_sub.csv')
age_submission = pd.read_csv('./data/age_sub.csv')
loc_submission = pd.read_csv('./data/location_sub.csv')
merge_submission = pd.merge(sex_submission,age_submission,how='left',on='uid')
merge_submission = pd.merge(merge_submission,loc_submission,how='left',on='uid')
merge_submission.to_csv('./data/merge_answer_v1.csv',index=False)
