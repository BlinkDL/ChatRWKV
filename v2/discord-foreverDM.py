from chat_foreverDM import user, bot, interface, on_message, out, srv
import pandas as pd
import random
import nest_asyncio
import interactions
nest_asyncio.apply()

ins = '''You are the DM for a game that's a cross between Dungeons and Dragons and a choose-your-own-adventure game. You will be given an action and a sentence about how that action goes. You will send me an immersive and detailed response describing how the action went for [char].
### Input:'''
# srv = 'dummy_server'
delim = '##########'
GUILD_ID = 'MTA5Nzk2ODA1OTA3NzA5OTU1MA.GrchrB.DNQ63eF5r5BbaHYGac7De6g2nEd9TtkNhijK68'

die=pd.read_csv('/root/foreverDM/newRollingTable.csv',index_col=None)



client = interactions.Client()

@interactions.listen()
async def on_ready():
    print(f"Logged in as {client.user} (ID: {client.user.id}), {client.status}")
    print("------")
    
@interactions.slash_command("dm", description="summon foreverDM!")
@interactions.slash_option(
    "msg",
    "send foreverDM a sentence with a clear action, and she'll tell you the outcome!",
    opt_type=interactions.OptionType.STRING,
    required=True)
async def dm(ctx: interactions.SlashContext,msg:str):
    await ctx.defer()
    srv = str(ctx.author.username)
    if '+' in msg:
        output = on_message(msg+'\r')
        if output == None:
            outout = 'Chat reset!'
        await ctx.send(f"{output}")
    else:
        r=random.randint(0,19)
        roll=die['sentence'][r]
        Omsg = msg
        msg = f'{user}{interface}{msg}'
        result = f'\nresult: {roll}'
        print(msg+result)
        output = on_message(ins+msg + result,FREE_GEN_LEN = 100)
        await ctx.send(f"Action: {Omsg}\n**Dice: {die['die'][r]}**\n{output}")
    # await ctx.send(f"You input {msg}")

client.change_presence('online')
client.start("MTA5Nzk2ODA1OTA3NzA5OTU1MA.GcX-6l.GiGkiaL7creZvi-VYQLAK2N2nZk2c6Lm85I_18")