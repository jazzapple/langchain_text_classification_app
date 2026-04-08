from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


# Help LLM out with some few shot examples. Currently model tends to predict Business where it is Sci/Tech
# The example_prompt template must match your examples dict keys (text, category, reason).
few_shot_examples = [ # from train dataset
  {
    "text": "Venezuelans Vote Early in Referendum on Chavez Rule (Reuters) Reuters - Venezuelans turned out early\\and in large numbers on Sunday to vote in a historic referendum\\that will either remove left-wing President Hugo Chavez from\\office or give him a new mandate to govern for the next two\\years.",
    "category": "World",
    "reason": "Reports on Venuzuelan elections, protests, politics and civic issues"
    },
 {
    "text": "S.Koreans Clash with Police on Iraq Troop Dispatch (Reuters) Reuters - South Korean police used water cannon in\\central Seoul Sunday to disperse at least 7,000 protesters\\urging the government to reverse a controversial decision to\\send more troops to Iraq.",
    "category": "World",
    "reason": "Focuses on protests in South Korea in relation to a government decision to participate in conflict in Iraq"},
 {
    "text": "Phelps, Thorpe Advance in 200 Freestyle (AP) AP - Michael Phelps took care of qualifying for the Olympic 200-meter freestyle semifinals Sunday, and then found out he had been added to the American team for the evening's 400 freestyle relay final. Phelps' rivals Ian Thorpe and Pieter van den Hoogenband and teammate Klete Keller were faster than the teenager in the 200 free preliminaries.",
    "category": "Sports",
    "reason": "Covers swimming competition results"
    },
 {
    "text": "Reds Knock Padres Out of Wild-Card Lead (AP) AP - Wily Mo Pena homered twice and drove in four runs, helping the Cincinnati Reds beat the San Diego Padres 11-5 on Saturday night. San Diego was knocked out of a share of the NL wild-card lead with the loss and Chicago's victory over Los Angeles earlier in the day.",
    "category": "Sports",
    "reason": "Describes results from a national baseball league game"
    },
 {
    "text": "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again.",
    "category": "Business",
    "reason": "Describes stock market movements on wall street stock exchange"
    },
 {
    "text": "Carlyle Looks Toward Commercial Aerospace (Reuters) Reuters - Private investment firm Carlyle Group,\\which has a reputation for making well-timed and occasionally\\controversial plays in the defense industry, has quietly placed\\its bets on another part of the market.",
     "category": "Business",
     "reason": "Discusses a private investment firm's investment in an aerospace company"
     },
 {
    "text": "Madden, ESPN Football Score in Different Ways (Reuters) Reuters - Was absenteeism a little high\\on Tuesday among the guys at the office? EA Sports would like\\to think it was because 'Madden NFL 2005' came out that day,\\and some fans of the football simulation are rabid enough to\\take a sick day to play it.",
    "category": "Sci/Tech",
    "reason": "Describes potential workpolace absenteeism due to a new video game, Madden NFL 2005, released by video game company EA Sports"
    },
 {
    "text": "Group to Propose New High-Speed Wireless Format (Reuters) Reuters - A group of technology companies\\including Texas Instruments Inc. (TXN.N), STMicroelectronics\\(STM.PA) and Broadcom Corp. (BRCM.O), on Thursday said they\\will propose a new wireless networking standard up to 10 times\\the speed of the current generation.",
    "category": "Sci/Tech",
    "reason": "Focuses on a group of technology companies proposing a new wireless networking standard"
    },
  ]
few_shot_example_prompt = ChatPromptTemplate.from_messages([
      ("human", "{text}"),
      ("ai", "{category} — {reason}"),
  ])                                                                                                                  
  
few_shot_prompt = FewShotChatMessagePromptTemplate(                                                                 
      examples=few_shot_examples,
      example_prompt=few_shot_example_prompt,
  ) 

classification_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an experienced newspaper editor who has experience working with content across news categories. "
        "Classify the article into exactly one of: World, Sports, Business, Sci/Tech. "
        "Respond with the category and a brief one-sentence reason."
    )),
    few_shot_prompt,
    ("human", "{text}"),
])

