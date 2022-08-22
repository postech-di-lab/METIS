ydata-ymusic-rating-study-v1_0

Yahoo! Music ratings for user selected and randomly selected songs, version 1.0

=====================================================================
This dataset is provided as part of the Yahoo! Research Alliance Webscope program, to be used for approved non-commercial research purposes by recipients who have signed a Data Sharing Agreement with Yahoo!. This dataset is not to be redistributed. No personally identifying information is available in this dataset. More information about the Yahoo! Research Alliance Webscope program is available at
http://research.yahoo.com
=====================================================================

Full description:

This dataset contains ratings for songs collected from two different sources. The first source consists of ratings supplied by users during normal interaction with Yahoo! Music services. The second source consists of ratings for randomly selected songs collected during an online survey conducted by Yahoo! Research and hosted by Yahoo! Music. Users of Yahoo! LaunchCast radio in the US were able participate in the survey by clicking a link in the LaunchCast player. During the survey, participants were asked several multiple choice questions regarding their own rating behavior. Participants were then asked to rate 10 songs selected at random from a fixed set of 1000 songs. Participants had the option of listening to a 30 second clip of each song as many times as they wanted before entering their ratings. They were also able to change their ratings multiple times before submitting the survey. The set of 1000 songs used in the survey was selected at random from all songs in the Yahoo! Music rating database with an available audio clip, and at 
least 500 existing ratings. 

The existing ratings for each survey participant were extracted from the database of song ratings maintained by Yahoo! Music. 5400 survey participants were randomly selected for inclusion in this data set on the condition that they had at least 10 existing ratings in the Yahoo! Music rating database restricted to the fixed set of 1000 songs used in the survey. An additional set of 10,000 users was randomly selected for inclusion in the data set from among all non-survey participants with at least 10 existing ratings in the Yahoo! Music rating database restricted to the fixed set of 1000 songs used in the survey.  The data set includes approximately 300,000 user supplied ratings and exactly 54,000 ratings for randomly selected songs. All users and items are represented by randomly assigned numeric identification numbers.  

In addition, the data set includes seven survey questions with responses for each of the 5400 survey participants included in the data set. The survey data and ratings for randomly selected songs were collected between August 22, 2006 and September 7, 2006. The existing rating data obtained from Yahoo! Music was collected between 2002 and 2006. 

This data set in intended for use in collaborative filtering and recommender systems research. The rating data has been divided into a training set, and a test set. The test set consists of the 54,000 ratings for randomly selected songs, while the test set consists of approximately 300,000 user-supplied ratings. Note that there is no test data for the the 10,000 non-survey participants, which are included for training purposes only. 

This dataset consists of four text files:
1. ydata-ymusic-rating-study-v1_0-train.txt
2. ydata-ymusic-rating-study-v1_0-test.txt
3. ydata-ymusic-rating-study-v1_0-survey-questions.txt
4. ydata-ymusic-rating-study-v1_0-survey-answers.txt


The content of the files are as follows:

=====================================================================

1. "ydata-ymusic-rating-study-v1_0-train.txt": The training data file contains data for 15,400 users and 1000 songs. There is a total of approximately 300,000 ratings. Each user has at least 10 observed ratings. Each user has at most one observation for each song. The users are ordered by
randomly assigned user id. The observations for each user are listed sequentially, and are ordered by randomly assigned song id. The first 5400 users are survey participants while the last 10,000 are the non-survey participants. The ratings values are on a scale from 1 to 5. The format of each row of each file is "user id<TAB>song id<TAB>rating".

Snippet:
1	14	5
1	35	1
1	46	1
1	83	1
1	93	1

====================================================================

2. "ydata-ymusic-rating-study-v1_0-test.txt": The test data file contains data for the 5,400 survey
participants and 1000 songs. There is a total of exactly 54,000 ratings. Each user has exactly 10 observed ratings. The users are ordered by randomly assigned user id. The observations for each user are listed sequentially, and are ordered by randomly assigned song id. The ratings values are on a scale from 1 to 5. The format of each row of each file is "user id<TAB>song id<TAB>rating".

Snippet:
1	49	1
1	126	1
1	138	1
1	141	1
1	177	1

====================================================================

3. "ydata-ymusic-rating-study-v1_0-survey-questions.txt": The seven survey questions a listed one per line. Each question is listed first, followed by the multiple choice answers. The questions and answers are separated by <TAB>'s.

Snippet:
If I hear a song I hate I choose to rate it:   never   very infrequently    infrequently   often   very often 
If I hear a song I don't like I choose to rate it:   never   very infrequently    infrequently   often   very often 


====================================================================

4. "ydata-ymusic-rating-study-v1_0-survey-answers.txt": This file contains the answers to the seven multiple choice survey questions for each of the 5400 survey participants. Line "n" of the file contains the answers to all of the multiple choice questions for user "n". The answers are separated by <TAB>'s. The answer for each question has been encoded using integers from 1 to the number of answers for that question. A value of "1" for the response to a given question corresponds the first answer listed in the file "ydata-ymusic-rating-study-v1_0-survey-questions.txt" for that question. A value of "2" corresponds to the second answer listed, etc.


Snippet:
5	5	4	3	5	5	2
5	4	4	4	5	5	1
5	5	5	5	5	5	1
5	5	5	5	5	5	2
5	5	5	5	5	5	2


