interface = ":"
user = "###Instruction"
bot = "###Response"

# If you modify this, make sure you have newlines between user and bot words too

init_prompt = f'''{user}{interface} You are the DM for a game that's a cross between Dungeons and Dragons and a choose-your-own-adventure game. You will be given an action and a sentence about how that action goes. You will send me an immersive and detailed response describing how the action went for [char].
###input: Doing a weighted one-arm box jump with a heavy weight
result: The action goes poorly, with many negative outcomes and limited positive outcomes.
{bot}{interface} As [char] prepared to do a weighted one-arm box jump with a heavy weight, they felt confident in their strength and abilities. However, as they began the exercise, they quickly realized they had underestimated the weight and difficulty of the jump. Despite their best efforts, they struggled to lift the weight and jump onto the box, resulting in several failed attempts. On one attempt, they lost their balance and fell off the box, injuring their ankle.
##########

{user}{interface} You are the DM for a game that's a cross between Dungeons and Dragons and a choose-your-own-adventure game. You will be given an action and a sentence about how that action goes. You will send me an immersive and detailed response describing how the action went for [char].
###input: Uncovering a hidden society of people with supernatural abilities and trying to understand their origins and purpose.
result: The action goes extremely well, exceeding expectations and achieving almost all desired outcomes.
{bot}{interface} As [char] delved deeper into the mystery of the hidden society of people with supernatural abilities, they found themselves uncovering more secrets and gaining a greater understanding of their origins and purpose. With each discovery, [char] became more and more fascinated by this world of wonder and intrigue. They spent countless hours studying ancient texts and artifacts, piecing together the history and culture of this society. As they gained the trust of some of the members, [char] was able to learn about their unique abilities and the ways in which they used them to help others. Through their interactions, [char] was able to form deep connections with these individuals and gain insight into their struggles and triumphs.
##########

{user}{interface} You are the DM for a game that's a cross between Dungeons and Dragons and a choose-your-own-adventure game. You will be given an action and a sentence about how that action goes. You will send me an immersive and detailed response describing how the action went for [char].
###input: Doing a weighted one-arm box jump with a heavy weight
result: The action goes poorly, with many negative outcomes and limited positive outcomes.
{bot}{interface} As [char] prepared to do a weighted one-arm box jump with a heavy weight, they felt confident in their strength and abilities. However, as they began the exercise, they quickly realized they had underestimated the weight and difficulty of the jump. Despite their best efforts, they struggled to lift the weight and jump onto the box, resulting in several failed attempts. On one attempt, they lost their balance and fell off the box, injuring their ankle.
##########

{bot}{interface} You are the DM for a game that's a cross between Dungeons and Dragons and a choose-your-own-adventure game. You will be given an action and a sentence about how that action goes. You will send me an immersive and detailed response describing how the action went for [char].
###input: Uncovering a hidden society of people with supernatural abilities and trying to understand their origins and purpose.
result: The action goes extremely well, exceeding expectations and achieving almost all desired outcomes.
{bot}{interface} As [char] delved deeper into the mystery of the hidden society of people with supernatural abilities, they found themselves uncovering more secrets and gaining a greater understanding of their origins and purpose. With each discovery, [char] became more and more fascinated by this world of wonder and intrigue. They spent countless hours studying ancient texts and artifacts, piecing together the history and culture of this society. As they gained the trust of some of the members, [char] was able to learn about their unique abilities and the ways in which they used them to help others. Through their interactions, [char] was able to form deep connections with these individuals and gain insight into their struggles and triumphs.
##########

{bot}{interface} You are the DM for a game that's a cross between Dungeons and Dragons and a choose-your-own-adventure game. You will be given an action and a sentence about how that action goes. You will send me an immersive and detailed response describing how the action went for [char].
###input: Describing the impact of the Silk Road on trade and cultural exchange between Europe and Asia.
result: The action goes poorly, with many negative outcomes and limited positive outcomes.
{bot}{interface} As [char] delved into the history of the Silk Road, they discovered that the impact on trade and cultural exchange between Europe and Asia was not entirely positive. While the Silk Road did facilitate the exchange of goods and ideas between the two regions, it also led to the spread of disease and conflict. The constant movement of people and goods along the Silk Road made it an ideal breeding ground for diseases, which were then spread to different parts of the world. Additionally, the Silk Road was often targeted by bandits and other criminals, making trade and travel dangerous and unpredictable. 
Furthermore, the Silk Road was not always an equal exchange, with some regions benefiting more than others.
'''