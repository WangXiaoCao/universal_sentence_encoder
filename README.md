# [Community Question Answering (CQA)](http://alt.qcri.org/semeval2017/task3/)

In essence, the main CQA task can be defined as follows:

“given (i) a new question and (ii) a large collection of question-comment threads created by a user community, rank  the comments that are most useful for answering the new question"

More details about community question answering can be found [here](http://alt.qcri.org/semeval2017/task3/).

This project specifically aims to address the following two sub-tasks.

### 1. Duplicate Question Detection

If two questions are pointing to the same issue but they are just written in different ways, how can we detect them?  Given the new question and a set of questions posted previously, identify if the new question is a duplicate of any previously posted question. An example is given below.

#### New question:
 
**Title**: Why do bread companies add sugar to bread?

**Body**: I have a client who is on a sugar detox/diet. She can't eat any bread because all the bread companies added sugar. Why do bread companies add sugar to their breads?

#### Duplicate question:
 
**Title**: What is the purpose of sugar in baking plain bread?

**Body**: My recipe says 1 tablespoon of sugar per loaf. This seems like too small an amount for flavor.
The recipe is as follows:
    3 cups flour
    1 teaspoon salt
    2 teaspoons active-dry yeast
    1 tablespoon sugar
    2 tablespoons oil
    1 cup water
    knead, wait 1 hr, knead again, wait 1.25 hr, bake for 30min @ 350
Is this for flavor, or is there another purpose?

#### Non-duplicate question:
 
**Title**: Is it safe to eat potatoes that have sprouted?

**Body**: I'm talking about potatoes that have gone somewhat soft and put out shoots about 10cm long. Other online discussions suggest it's reasonably safe and the majority of us have been peeling and eating soft sprouty spuds for years. Is this correct?

### 2. Duplicate Answer Detection
Same as duplicate question detection task.
