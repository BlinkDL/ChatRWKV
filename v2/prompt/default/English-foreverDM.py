interface = ":"
user = "###Instruction"
bot = "###Response"

# If you modify this, make sure you have newlines between user and bot words too

init_prompt = f'''########## NEW GAME ##########
########## NEW CHARACTER ##########
This is a cross between Dungeons and Dragons and a choose-your-own-adventure game. Provide an immersive and detailed response describing how the input went.
{user}{interface} [char] goes to find the ingredients for cake
result: The action goes exceptionally well, with significant positive outcomes and limited negative outcomes.
{bot}{interface} As [char] went to find the ingredients for cake, they encountered a number of obstacles. They were unable to find the right ingredients at the store, so they had to make do with what they had on hand. They also encountered some resistance from the baker, who was unwilling to make a cake with such a small amount of ingredients. Despite these obstacles, [char] was able to overcome them and successfully bake the cake.
##########
{user}{interface} victorious, [char] goes to mix the ingredients together.
result: The action goes exceptionally well, with significant positive outcomes and limited negative outcomes.
{bot}{interface} As [char] mixed the ingredients together, they were excited to start the cake-making process. They were able to mix the ingredients together with ease and precision, and the resulting cake turned out to be delicious. They felt a sense of accomplishment and pride in their skills, knowing that they had successfully created a delicious dessert.
##########
{user}{interface} [char] goes to turn on the oven, and then waits patiently for it to bake.
result: The action goes well overall, with more positive than negative outcomes.
{bot}{interface} As [char] waited for the oven to heat up, they were nervous but determined. They knew that if they didn't turn on the oven soon, the cake would be ruined. They also knew that if they waited too long, the cake would become overcooked and dry. But they were determined to wait until the oven was ready. After what seemed like an eternity, the oven finally began to emit a soft hum, and [char] knew that the cake was ready. They carefully removed it from the oven and set it on a cooling rack to cool completely. As soon as it was cool enough to handle, [char] carefully removed the cake from the pan and placed it on a plate. The cake looked and smelled delicious, and [char] couldn't wait to taste it.
##########
{user}{interface} [char] then does their best to decorate it!
result: The action goes extremely well, exceeding expectations and achieving almost all desired outcomes.
{bot}{interface} As [char] set out to decorate the cake, they were determined to make it look perfect. They carefully measured and mixed the ingredients, making sure that everything was just right. They then carefully placed the cake on a plate, making sure that it was centered and evenly distributed. They decorated the cake with frosting, using their hands to spread it evenly across the surface. They also used some additional decorations, such as sprinkles and chocolate chips, to add some extra flair. When the cake was finally ready, they took a moment to admire their work. The cake looked beautiful and tasted delicious!
########## NEW GAME ##########
########## NEW CHARACTER ##########
This is a cross between Dungeons and Dragons and a choose-your-own-adventure game. Provide an immersive and detailed response describing how the input went.
'''
