#!/usr/bin/env python3

import os
import discord
from discord.ext import commands
from discord import Message, Embed, Interaction, File
from dotenv import load_dotenv
from groq import Groq
from collections import defaultdict
import requests
import json
from datetime import datetime, timedelta
import google.generativeai as gemini
import textwrap
import traceback
import asyncio
import logging
from bs4 import BeautifulSoup 
from hunger_games import HungerGames, Participant 
import random
import edge_tts
import io
from contextlib import redirect_stdout
import aiohttp
import lyricsgenius
from timeit import default_timer as timer 
from typing import Any
import json
from PIL import Image
from PIL.ExifTags import TAGS
import yt_dlp
# Load environment variables from .env file
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
MVSEP_API_KEY = os.getenv('MVSEP_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') 
GENIUS_API_TOKEN = os.getenv('GENIUS_API_TOKEN')
# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Initialize the Google Generative AI client
gemini.configure(api_key=GOOGLE_API_KEY)

# Initialize genius shit
genius = lyricsgenius.Genius("GENIUS_API_TOKEN")

# Initialize the bot with intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Authorized users/roles (replace with actual IDs)
authorized_users = [936673139419664414]
authorized_roles = [1198707036070871102] 

# responses to on message

# Bot-wide settings
bot_settings = {
    "model": "llama3-70b-8192",
    "system_prompt": "You are a helpful and friendly AI assistant.",
    "context_messages": 5,
    "llm_enabled": False 
}
code_language = "python"
# Define valid model names for Groq and Gemini
groq_models = [
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma-7b-it",
    "mixtral-8x7b-32768"
]

# Define valid model names for Gemini
gemini_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro-2m-latest" 
]


    # --- Conversation Data (Important!) chatting shit
conversation_data = defaultdict(lambda: {"messages": []}) 

# --- loggin shit
LOG_CHANNEL_ID = 1251625595431813144
logging.basicConfig(filename='bot_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


# --- Helper Function for Checking Authorization ---

def is_authorized(interaction: discord.Interaction):
    """Check if the user has permission to use the command."""
    user = interaction.user
    if user.id in authorized_users:
        return True
    if any(role.id in authorized_roles for role in user.roles):
        return True
#    if user.id == interaction.guild.owner_id:
#        return TrueF
    return False

# --- Application Commands ---
@bot.tree.command(name="eval", description="Evaluate Python code. )")
async def eval_code(interaction: discord.Interaction, code: str):
    """
    Evaluates Python code provided by the bot owner. 

    Features:
     - Code block formatting
     - Standard output capture
     - Error handling and traceback display
     - Result truncation for large outputs
     - Execution time measurement
    """

    if not is_authorized(interaction):
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    env = {
        "bot": bot,
        "discord": discord,
        "commands": commands,
        "interaction": interaction,
        "channel": interaction.channel,
        "guild": interaction.guild,
        "message": interaction.message,
    }

    code = code.strip("`")

    stdout = io.StringIO()

    start_time = timer() 
    try:
        with redirect_stdout(stdout):
            exec(
                f"async def func():\n{textwrap.indent(code, '    ')}", env
            )
            result = await env["func"]()  
            result_str = str(result) if result is not None else "No output."

    except Exception as e:

        result_str = "".join(
            traceback.format_exception(type(e), e, e.__traceback__)
        )

    end_time = timer() 
    execution_time = (end_time - start_time) * 1000 


    if len(result_str) > 1900: 
        result_str = result_str[:1900] + "... (Output truncated)"


    await interaction.response.send_message(
        f"```python\n{code}\n```\n**Output:**\n```\n{result_str}\n```\n**Execution time:** {execution_time:.2f} ms",
        ephemeral=True,
    )
@bot.command(name="lyrics", description="Search for song lyrics.")
async def lyrics(ctx, *, song_title: str):
    async with ctx.typing():
        try:
            search_results = genius.search_songs(song_title)

            if not search_results:
                await ctx.send(f"No lyrics found for '{song_title}'.")
                return
            best_match = search_results[0] 
            for result in search_results:
                if song_title.lower() == result.title.lower():
                    best_match = result
                    brea

            song = genius.song(best_match.id) 
            lyrics_text = song.lyrics
            if len(lyrics_text) > 2000:  
                lyrics_text = lyrics_text[:2000] + "... (Lyrics truncated)"
            
            await ctx.send(f"```\n{lyrics_text}\n```") 

        except Exception as e:
            await ctx.send(f"An error occurred while fetching lyrics: {e}")
            print(f"Error fetching lyrics: {e}")

MAX_AUDIO_SIZE = 15 * 60 * 1024 * 1024  

@bot.tree.command(name="separate", description="Separate uploaded audio into its components")
async def separate(interaction: discord.Interaction, audio_file: discord.Attachment):

    if audio_file.size > MAX_AUDIO_SIZE:
        logging.error(f"User: {interaction.user} - Error: audio file over 15 minutes")
        await interaction.response.send_message("Sorry, audio files must be under 15 minutes long.", ephemeral=True)
        return
    logging.info(f"User: {interaction.user} - seperated audio")
    await interaction.response.send_message("Separating audio... This might take a moment.", ephemeral=True)

    try:
        file_path = f"mvsep/{audio_file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        await audio_file.save(file_path)
        async with aiohttp.ClientSession() as session:
            with open(file_path, 'rb') as f:
                data = {
                    'api_token': MVSEP_API_KEY,
                    'sep_type': '40',
                    'add_opt1': '5', 
                    'audiofile': f,
                    'output_format': "1"
                }
                async with session.post("https://mvsep.com/api/separation/create", headers={'Authorization': f'Bearer {MVSEP_API_KEY}'}, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        job_hash = result['data']['hash']

                        while True:
                            await asyncio.sleep(5)
                            async with session.get(f'https://mvsep.com/api/separation/get?hash={job_hash}') as status_response:
                                if status_response.status == 200:
                                    status_result = await status_response.json()
                                    if status_result['status'] == 'done':
                                        separated_files = status_result['data']['files']
                                        urls = [file['url'] for file in separated_files]
                                        await interaction.user.send(f"Audio separated successfully! Download them here:\n" + "\n".join(urls))
                                        logging.info(f"User: {interaction.user} - seperated audio - audio urls: {urls}")
                                        break
                                    elif status_result['status'] == 'failed':
                                        await interaction.user.send(f"Audio separation failed: {status_result['data']['message']}")
                                        logging.error(f"User: {interaction.user} - Error: {status_result['data']['message']} ")
                                        break
                                else:
                                    await interaction.user.send(f"Failed to check status. Status code: {status_response.status}")
                                    logging.error(f"User: {interaction.user} - Error: {status_response.status} ")
                                    break
                    else:
                        await interaction.user.send(f"Failed to separate audio. Status code: {response.status}")
        os.remove(file_path)
    except aiohttp.ClientConnectorError as e:
        await interaction.user.send(f"Error connecting to the MVSEP API: {e}")
    except Exception as e:
        await interaction.user.send(f"An error occurred: {e}")
        
@bot.tree.command(name="kill", description="murder")
async def kill(interaction: discord.Interaction, user: str):
  """
  murder
  """

  await interaction.response.send_message(f"{user} was killed")

import asyncio
from datetime import datetime, timedelta


@bot.tree.command(name="reminder", description="Set a reminder.")
async def reminder(interaction: discord.Interaction, time: str, *, message: str):
    """
    Set a reminder.

    Usage: /reminder <time> <message>

    Time format:
    - 5s (seconds)
    - 10m (minutes)
    - 1h (hour)
    - 2d (days)

    Example: /reminder 1h Study for the exam
    """
    time_unit = time[-1]
    time_amount = int(time[:-1])
    if time_unit == "s":
        delta = timedelta(seconds=time_amount)
    elif time_unit == "m":
        delta = timedelta(minutes=time_amount)
    elif time_unit == "h":
        delta = timedelta(hours=time_amount)
    elif time_unit == "d":
        delta = timedelta(days=time_amount)
    else:
        await interaction.response.send_message("Invalid time format. Use s/m/h/d for seconds/minutes/hours/days.", ephemeral=True)
        logging.error(f"User: {interaction.user} - Error: Invalid time format. Use s/m/h/d for seconds/minutes/hours/days")
        return

    await interaction.response.send_message(f"Reminder set for {time} from now.") 
    logging.info(f"User:{interaction.user} - set a timer for {time}")

    await asyncio.sleep(delta.total_seconds())
    await interaction.channel.send(f"<@{interaction.user.id}> ⏰ Reminder: {message}") 
    logging.info(f"User:{interaction.user} - Reminder: {message}")

@bot.tree.command(name="hunger_games", description="Start a Hunger Games simulation with Discord users.")
async def hunger_games(interaction: discord.Interaction, *, users: str):
    """Start a Hunger Games simulation.
    Mention Discord users separated by spaces. Example:
    /hunger_games @user1 @user2 @user3 @user4
    """


    await interaction.response.send_message(f"Gathering tributes... This might take a moment.")


    mentioned_users = interaction.data['resolved']['members'].values()

    if len(mentioned_users) < 2:
        await interaction.followup.send("Please mention at least two Discord users to participate.")
        return


    participants = [Participant(user['user']['username'], user['user']['id'], user['user']['avatar']) for user in mentioned_users]

    game = HungerGames(participants)
    await interaction.followup.send("Let the games begin!")


    round_number = 1
    while len(game.participants) > 1:
        embed = Embed(title=f"🔥 The Hunger Games - Round {round_number} 🔥", color=discord.Color.red())
        round_messages = []

        for participant in game.participants[:]:
            if len(game.participants) <= 1:
                break

            scenario = game.choose_scenario(participant) 
            if scenario in [
            game.kill_scenario, game.form_alliance_scenario,
            game.betrayal_scenario, game.steal_supplies_scenario,
            game.item_kill_scenario, game.sleeping_scenario,
            game.help_scenario, game.trap_scenario, game.mutual_rescue_scenario
            ]:
                valid_others = [p for p in game.participants if p != participant]
                if valid_others:
                    other = random.choice(valid_others)
                    output = io.StringIO()
                    with redirect_stdout(output):
                        participant.interact(scenario, [other])

                    user = interaction.guild.get_member(participant.user_id)
                    if user:
                        round_messages.append(f"{output.getvalue().strip()}")
                        embed.set_thumbnail(url=user.avatar.url)
                    else:
                        round_messages.append(f"{output.getvalue().strip()}")
            else:
                output = io.StringIO()
                with redirect_stdout(output):
                    participant.interact(scenario)

                user = interaction.guild.get_member(participant.user_id)
                if user:
                    round_messages.append(f"{output.getvalue().strip()}")
                    embed.set_thumbnail(url=user.avatar.url)
                else:
                    round_messages.append(f"{output.getvalue().strip()}")

        embed.description = "\n\n".join(round_messages)
        await interaction.channel.send(embed=embed)

        await interaction.channel.send("Type 'next' to continue...")
        def check(m):
            return m.author == interaction.user and m.channel == interaction.channel and m.content.lower() == 'next'
        try:
            await bot.wait_for('message', check=check, timeout=360)
        except asyncio.TimeoutError:
            await interaction.channel.send("The game has timed out due to inactivity.")
            return

        round_number += 1

    await interaction.channel.send("The Hunger Games have ended!")
    if game.participants:
        winner = game.participants[0]
        embed = Embed(title="🏆 The Victor 🏆", color=discord.Color.gold())
        embed.description = f"{winner.name} has won the Hunger Games!"

        winner_user = interaction.guild.get_member(winner.user_id) 
        if winner_user:
            embed.set_thumbnail(url=winner_user.avatar.url) 

        await interaction.channel.send(embed=embed)
    else:
        await interaction.channel.send("There are no survivors.")

@bot.tree.command(name="serverinfo", description="Get information about the server.")
async def serverinfo(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    embed = discord.Embed(
        title="Sanctuary: Open Source AI",
        description="Sanctuary is a Discord server dedicated to open-source AI projects and research. It's a place for users, developers, and researchers to connect, share their work, help each other and collaborate.  The server aims to highlight amazing open-source projects and inspire developers to push the boundaries of AI.",
        color=discord.Color.blue()
    )
    embed.add_field(
        name="How to Help",
        value="1. Boost the server to unlock more features.\n2. Spread the word to your friends.\n3. Help improve the server by posting suggestions in the designated channel.",
        inline=False
    )
    embed.add_field(name="Permanent Invite Link", value="[Join Sanctuary](https://discord.gg/kSaydjBXwf)", inline=False)
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="set_model", description="Set the language model for the entire bot.")
async def set_model(interaction: discord.Interaction, model_name: str):
    if not is_authorized(interaction):
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    model_name = model_name.lower()
    if model_name in groq_models:
        bot_settings["model"] = model_name
        await interaction.response.send_message(f"Model set to: **{model_name}** for the entire bot.")
    elif model_name in gemini_models:
        bot_settings["model"] = model_name
        await interaction.response.send_message(f"Model set to: **Google Gemini {model_name}** for the entire bot.")
    else:
        await interaction.response.send_message(f"Invalid model.\nAvailable models:\nGroq: {', '.join(groq_models)}\nGemini: {', '.join(gemini_models)}")

@bot.tree.command(name="search_github_projects", description="Search for GitHub projects.")
async def search_github_projects(interaction: discord.Interaction, query: str):
    """Search for GitHub projects based on a search query.

    Args:
        query: The GitHub search query (e.g., 'machine learning', 'topic:natural-language-processing').
    """
    try:

        url = "https://api.github.com/search/repositories"
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": 1 
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        matching_repos = response.json()["items"]

        if matching_repos:
            embed = discord.Embed(
                title=f"GitHub Project Search Results for: {query}",
                color=discord.Color.green() 
            )

            for repo in matching_repos:
                repo_name = repo['name']
                repo_url = repo['html_url']
                description = repo['description'] or "No description."

                embed.add_field(
                    name=f"{repo_name}",
                    value=f"**[Link to Repo]({repo_url})**\n{description}\n"
                          f"⭐ {repo['stargazers_count']}   "
                          f"🍴 {repo['forks_count']}",
                    inline=False
                )

            await interaction.response.send_message(embed=embed)
        else:
            await interaction.response.send_message(f"No projects found for query: {query}")

    except requests.exceptions.RequestException as e:
        await interaction.response.send_message(f"An error occurred while searching GitHub: {e}")

@bot.tree.command(name="help", description="Show available commands.")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(title="Available Commands", color=discord.Color.blue())

    embed.add_field(name="/serverinfo", value="Gives info about the server.", inline=False)
    embed.add_field(name="/set_model <model_name>", value="Set the language model for the entire bot. (ADMIN)", inline=False)
    embed.add_field(name="/set_system_prompt <prompt>", value="Set the system prompt for the entire bot. (ADMIN)", inline=False)
    embed.add_field(name="/set_context_messages <num_messages>", value="Set the number of context messages to use (1-10) for the entire bot. (ADMIN)", inline=False)
    embed.add_field(name="/say <message>", value="Make the bot say something. (ADMIN)", inline=False)
    embed.add_field(name="/toggle_llm", value="Turn the LLM part of the bot on or off for the entire bot. (ADMIN)", inline=False)
    embed.add_field(name="/trending_projects <query>", value="Show trending GitHub projects (past 7 days). Default query: 'topic:language-model'.", inline=False)
    embed.add_field(name="/search_github_projects <query>", value="Search for GitHub projects.", inline=False)
    embed.add_field(name="/summarize <prompt>", value="Summarize info given.", inline=False)
    embed.add_field(name="/play_audio", value="plays an audio based on what the file path in code says (ADMIN)", inline=False)
    await interaction.response.send_message(embed=embed)

@bot.tree.command(name="trending_projects", description="Show trending GitHub projects.")
async def trending_projects(interaction: discord.Interaction, query: str = "topic:language-model"):
    """Show trending GitHub projects based on a search query. 

    Args:
        query: The GitHub search query (e.g., 'topic:machine-learning'). 
               Defaults to 'topic:language-model'.
    """
    try:
        url = "https://api.github.com/search/repositories"
        date_threshold = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        params = {
            "q": f"{query} created:>{date_threshold}",
            "sort": "stars",
            "order": "desc",
            "per_page": 5
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        trending_repos = response.json()["items"]

        if trending_repos:
            embed = discord.Embed(
                title=f"Trending GitHub Projects for Query: {query} (Past 7 Days)",
                color=discord.Color.blue()
            )

            for repo in trending_repos:
                repo_name = repo['name']
                repo_url = repo['html_url']
                description = repo['description'] or "No description."

                field_value = f"{description}\n"
                field_value += f"⭐ {repo['stargazers_count']}   "
                field_value += f"🍴 {repo['forks_count']}"


                embed.add_field(
                        name=f"{repo_name}",
                        value=f"**[Link to Repo]({repo_url})**\n{description}\n"
                              f"⭐ {repo['stargazers_count']}   "
                              f"🍴 {repo['forks_count']}",
                        inline=False 
                    )
                    
            await interaction.response.send_message(embed=embed)
        else:
            await interaction.response.send_message(f"No trending projects found for query: {query}")

    except requests.exceptions.RequestException as e:
        await interaction.response.send_message(f"An error occurred while fetching data from GitHub: {e}")

@bot.tree.command(name="set_system_prompt", description="Set the system prompt for the entire bot.")
async def set_system_prompt(interaction: discord.Interaction, prompt: str):
    if not is_authorized(interaction):
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    bot_settings["system_prompt"] = prompt
    await interaction.response.send_message(f"System prompt set to:\n```\n{prompt}\n``` for the entire bot.")


@bot.tree.command(name="speak")
async def speak(interaction: discord.Interaction, text: str):
    """Speaks the given text in the user's voice channel."""
    if not is_authorized(interaction):
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return
    if interaction.user.voice is None:
        await interaction.response.send_message("You are not connected to a voice channel.", ephemeral=True)
        return
    voice_channel = interaction.user.voice.channel
    if interaction.guild.voice_client is None:
        await voice_channel.connect() 
    vc = interaction.guild.voice_client 
    try:
        tts = edge_tts.Communicate(text, "en-US-JennyNeural")

        if not os.path.exists("temp"):
            os.makedirs("temp")

        await tts.save("temp/tts.mp3") 

        source = discord.FFmpegPCMAudio("temp/tts.mp3")
        vc.play(source, after=lambda e: print(f'Finished playing: {e}'))

        while vc.is_playing():
            await asyncio.sleep(1) 

    except Exception as e:
        await interaction.response.send_message(f"An error occurred: {e}", ephemeral=True)
        return

@bot.tree.command(name="control_my_computer", description="Write and run code using an LLM (Admin Only).")
async def create_code(interaction: discord.Interaction, code_request: str):
    if not is_authorized(interaction):
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return
    try:
        prompt = f"""Write a {code_language} code snippet that will create and run: {code_request}
        the computer is windows 11
        The code should be executable directly. 
        Do not include any backticks or language identifiers in the output.
        have the code by itself with NO explanation
        never explain the code or give anything that is not just the pure code
        do not give any extra info
        name for windows user is user1 for file paths
        """
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192"
        )
        generated_code = chat_completion.choices[0].message.content
        generated_code = generated_code.strip("`") 
        generated_code_lobotomised = generated_code[:100] 
        await interaction.channel.send(f"prompt: {code_request}")
        logging.info(f"User: {interaction.user} - Prompt: {code_request} - Generated Code: {generated_code}")
        result = await execute_code(generated_code, code_language)
        if result == "No output.":
            await interaction.channel.send("Script ran")
            logging.info(f"User: {interaction.user} - {result}")
        else:
            await interaction.channel.send(f"result:{result}")
            logging.info(f"User: {interaction.user} - Code Output: {result}")
    except Exception as e:
        await interaction.response.send_message(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")

async def execute_code(code: str, language: str) -> str:
    try:
        if language == "python":
            result = await run_python_code(code)
        else:
            result = f"Execution for {language} is not supported yet."
        return result
    except Exception as e:
        print(f"An error occurred during code execution: {e}")
        return str(e)

async def run_python_code(code: str) -> str:
    try:
        with open("temp_code.py", "w") as f:
            f.write(code)

        proc = await asyncio.create_subprocess_exec(
            "python", "temp_code.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

        os.remove("temp_code.py")

        if stdout:
            return stdout.decode()
        if stderr:
            return stderr.decode()
        return "No output."

    except asyncio.TimeoutError:
        return "Code execution timed out."
    except Exception as e:
        return str(e)      


@bot.tree.command(name="summarize", description="Summarize a text using the current LLM.")
async def summarize(interaction: discord.Interaction, text: str):
    message = interaction.user
    try:
        selected_model = bot_settings["model"]

        if selected_model in gemini_models:
            try:
                gemini_model = gemini.GenerativeModel(selected_model) 

                response = gemini_model.generate_content( 
                    f"Summarize the following text:\n\n{text}",
                )

                summary = response.text
                await interaction.response.send_message(f"Summary:\n```\n{summary}\n```")
            except Exception as e:
                await interaction.response.send_message(f"An error occurred while processing the request: {e}")

        else:
            system_prompt = bot_settings["system_prompt"]
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
                ],
                model=selected_model
            )
            summary = chat_completion.choices[0].message.content

            logging.info(f"User: {message} - Model: {selected_model} - Summary: {summary}")

            await interaction.response.send_message(f"Summary:\n```\n{summary}\n```")

    except Exception as e:
        await interaction.response.send_message(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")


@bot.tree.command(name="ping", description="Get the bot's latency.")
async def ping(interaction: discord.Interaction):
    latency = bot.latency * 1000 
    await interaction.response.send_message(f"Pong! Latency is {latency:.2f} ms")

@bot.tree.command(name="summarize_website", description="summarize a website.")
async def summarize_website(interaction: discord.Interaction, website_url: str):
#    if not is_authorized(interaction):
#        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
#        return
    await interaction.response.defer() 
    try:
        response = requests.get(website_url)
        response.raise_for_status() 

        soup = BeautifulSoup(response.content, 'html.parser')

        extracted_text = ""
        for paragraph in soup.find_all('p'):
            extracted_text += paragraph.get_text() + "\n"

        if not extracted_text.strip(): 
            await interaction.response.send_message(content="Error: No text found on the website.") 
            return 

        selected_model = bot_settings["model"]
        if extracted_text == None:
         if selected_model in gemini_models:
            try:

                gemini_model = gemini.GenerativeModel(selected_model) 

                response = gemini_model.generate_content( 
                    f"Summarize the following text:\n\n{extracted_text}",
                )


                summary = response.text
                await interaction.response.send_message(f"Summary:\n```\n{summary}\n```")
            except Exception as e:
                await interaction.response.send_message(f"An error occurred while processing the request: {e}")

        else:
            lobotomised_extracted_text = extracted_text[:10000] 
            system_prompt = bot_settings["system_prompt"]
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarize the following text in detail and only output the summary itself for example do not add things 'like this is a summary':\n\n{lobotomised_extracted_text}"}
                ],
                model="llama3-8b-8192"
            )
            summary = chat_completion.choices[0].message.content
            


            logging.info(f"User: {interaction.user} - Website: {website_url} - Model: {selected_model} extracted text: {extracted_text} - Summary: {summary}")
            lobotomised_summary = summary[:1900]
            await interaction.followup.send(f"Summary of <{website_url}>:\n```\n{lobotomised_summary}\n```")

    except requests.exceptions.RequestException as e:
        await interaction.response.send_message(f"An error occurred while fetching the website: {e}")
    except Exception as e:
        await interaction.response.send_message(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")


@bot.tree.command(name="dm", description="Send a direct message to a user. (Authorized users only)")
async def dm(interaction: discord.Interaction, user: discord.User, *, message: str):
    """
    Sends a direct message to a specified user

    Usage: /dm <user> <message>

    Example: /dm @username hello there
    """
    if not is_authorized(interaction):
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    try:
        await user.send(message)
        await interaction.response.send_message(f"Message sent to {user.mention} successfully.", ephemeral=True)
        log_channel = bot.get_channel(LOG_CHANNEL_ID)
        if log_channel:
            embed = discord.Embed(
                title="Direct Message Sent",
                color=discord.Color.blue() 
            )
            embed.add_field(name="From", value=interaction.user.mention, inline=False)
            embed.add_field(name="To", value=user.mention, inline=False)
            embed.add_field(name="Message", value=message, inline=False)
            await log_channel.send(embed=embed)
        else:
            print(f"WARNING: Log channel with ID {LOG_CHANNEL_ID} not found.")

    except discord.HTTPException as e:
        await interaction.response.send_message(f"Failed to send message: {e}", ephemeral=True)

        log_channel = bot.get_channel(LOG_CHANNEL_ID)
        if log_channel:
            embed = discord.Embed(
                title="Direct Message Failed",
                color=discord.Color.red() 
            )
            embed.add_field(name="From", value=interaction.user.mention, inline=False)
            embed.add_field(name="To", value=user.mention, inline=False)
            embed.add_field(name="Error", value=e, inline=False)
            await log_channel.send(embed=embed)
        else:
            print(f"WARNING: Log channel with ID {LOG_CHANNEL_ID} not found.")

# ytdlp options 
ytdlp_opts = {
    'format': 'bestaudio/best',
    'extract-audio': True,  
    'noplaylist': True,
    'audio-format': 'mp3',  
    'outtmpl': 'yt/%(id)s.%(ext)s',  
    'postprocessors': [{  
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '128',
    }],
}
audio_queue = [] 
loop_enabled = False  



@bot.tree.command(name="play", description="Play audio from a YouTube link or search YouTube.")
async def play(interaction: discord.Interaction, *, query: str):
    if not is_authorized(interaction):
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    if interaction.user.voice is None:
        await interaction.response.send_message("You need to be connected to a voice channel.", ephemeral=True)
        return

    await interaction.response.defer()
    voice_channel = interaction.user.voice.channel
    vc = interaction.guild.voice_client

    if vc is None:
        if voice_channel.permissions_for(interaction.guild.me).connect:
            vc = await voice_channel.connect()
        else:
            await interaction.followup.send("I don't have permission to join that channel!", ephemeral=True)
            return

    try:
        if not query.startswith(('https://', 'http://')):
            query = f"ytsearch:{query}"

        with yt_dlp.YoutubeDL(ytdlp_opts) as ydl:
            info = ydl.extract_info(query, download=True)
            if 'entries' in info:
                info = info['entries'][0]
            audio_file = ydl.prepare_filename(info).replace('.webm', '.mp3')

            while not os.path.exists(audio_file):
                await asyncio.sleep(1)

            audio_queue.append(audio_file)
        print(f"Added to queue: {audio_file}")

        if not vc.is_playing():
            print("Starting playback from queue...")
            await play_next(interaction, vc)
            await interaction.followup.send(f"Now playing: {info['title']}")

    except Exception as e:
        await interaction.followup.send(f"An error occurred: {e}", ephemeral=True)
        print(f"Full error traceback:\n{traceback.format_exc()}")

async def play_next(interaction: discord.Interaction, vc: discord.VoiceClient):
    global loop_enabled, audio_queue

    if audio_queue:
        audio_file = audio_queue.pop(0)
        print(f"Playing from queue: {audio_file}")

        if not os.path.isfile(audio_file):
            print(f"Audio file not found: {audio_file}")
            return

        def after_playing(e):
            if loop_enabled:
                audio_queue.append(audio_file)
            else:
                if os.path.exists(audio_file):
                    try:
                        os.remove(audio_file)
                        print(f"Deleted: {audio_file}")
                    except Exception as e:
                        print(f"Error deleting file: {e}")

            asyncio.run_coroutine_threadsafe(play_next(interaction, vc), bot.loop)

        source = discord.FFmpegPCMAudio(audio_file)
        vc.play(source, after=lambda e: after_playing(e))

@bot.tree.command(name="loop", description="Toggle audio loop mode.")
async def loop(interaction: discord.Interaction):
    global loop_enabled
    loop_enabled = not loop_enabled
    await interaction.response.send_message(f"Loop mode is now {'enabled' if loop_enabled else 'disabled'}.")




@bot.tree.command(name="set_context_messages", description="Set the number of context messages to use (1-10) for the entire bot.")
async def set_context_messages(interaction: discord.Interaction, num_messages: int):
    if not is_authorized(interaction):
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    if 1 <= num_messages <= 100000:
        bot_settings["context_messages"] = num_messages
        await interaction.response.send_message(f"Number of context messages set to: {num_messages} for the entire bot.")
    else:
        await interaction.response.send_message("Invalid number of messages. Please choose between 1 and 10.")

@bot.tree.command(name="say", description="Make the bot say something.")
async def say(interaction: discord.Interaction, message: str):
    if not is_authorized(interaction):
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True) 
    await interaction.delete_original_response()

    await interaction.channel.send(message)


@bot.tree.command(name="toggle_llm", description="Turn the LLM part of the bot on or off for the entire bot.")
async def toggle_llm(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    bot_settings["llm_enabled"] = not bot_settings["llm_enabled"]
    new_state = "OFF" if not bot_settings["llm_enabled"] else "ON"
    await interaction.response.send_message(f"LLM is now turned {new_state} for the entire bot.")


@bot.tree.command(name="show_log", description="Send the last 2000 characters of the bot log.")
async def show_log(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        logging.error(f"{interaction.user} blocked from using show_log") 
        return

    try:
        with open('bot_log.txt', 'r') as log_file:
            log_file.seek(0, 2) 
            file_size = log_file.tell()
            offset = max(0, file_size - 2000)
            log_file.seek(offset)
            log_content = log_file.read()
            
            if len(log_content) == 0:
                await interaction.response.send_message("The log file is empty.")
            else:
                await interaction.response.send_message(f"```{log_content[-2000:]}```")
    except Exception as e:
        await interaction.response.send_message(f"An error occurred while reading the log file: {e}")
        logging.error(f"An error occurred while reading the log file: {e}") 


 --- Message Handling --- 

nuhuh = ["adfjhaskjfhaksfhjksa"]

nuhuh_responses = {
    "adfjhaskjfhaksfhjksa": "nuhuh",
}

ignored_users = [12, 123] 
message_reaction_cooldowns = {}
word_emoji_reactions = {
    "dflkajdl": "🔥",
    "dfhkjashf": "askdfhksj", 
    "skajdhfkjsad": "ksadjfhkjsad"   

}
word_message_reactions = {
    "falkdfj": "asdlkfjalsk",
    "alkdsfjkla": "haskdljfalkdfj" 
}

@bot.event
async def on_message(message: Message):
    if isinstance(message.channel, discord.DMChannel) and message.author != bot.user:
        global log_channel 
        log_channel = bot.get_channel(LOG_CHANNEL_ID) 
        embed = discord.Embed(title="Direct Message Received", color=discord.Color.blue())
        embed.add_field(name="From", value=f"{message.author.mention} ({message.author.id})", inline=False)
        embed.add_field(name="Message", value=message.content, inline=False)
        await log_channel.send(embed=embed)
    await bot.process_commands(message)
    if isinstance(message.channel, discord.TextChannel): 

        for word in nuhuh:
            if word.lower() in message.content.lower():
                try:
                    await message.delete()
                    await message.channel.send(nuhuh_responses[word], reference=message.reference)  
                    logging.warning(f"Deleted message from {message.author} containing '{word}': {message.content}") 
                except discord.Forbidden:
                    print("Missing permissions to delete message.")
                except discord.HTTPException as e:
                    print(f"Failed to delete message: {e}")
                break

    if message.author != bot.user: 
        for word, emoji in word_emoji_reactions.items():
            if word.lower() in message.content.lower():
                await message.add_reaction(emoji)
                break 


        for word, response in word_message_reactions.items():
            if word.lower() in message.content.lower():
                await message.channel.send(response)
                break  

    await bot.process_commands(message)
            
    is_mentioned = bot.user.mentioned_in(message)
    is_reply_to_bot = message.reference is not None and message.reference.resolved.author == bot.user

    if bot_settings["llm_enabled"] and (is_mentioned or is_reply_to_bot): 
        try:
            channel_id = str(message.channel.id)
            messages = conversation_data[channel_id]["messages"]  
            selected_model = bot_settings["model"]              
            system_prompt = bot_settings["system_prompt"]      
            context_messages_num = bot_settings["context_messages"] 

            context_messages = messages[-context_messages_num:]
            api_messages = [{"role": "system", "content": system_prompt}] + context_messages + [{"role": "user", "content": message.content}]
            lobotomised_generated_text = ""
            if selected_model in gemini_models:
                gemini_model_mapping = {
                    "flash": "gemini-1.5-pro-flash",
                    "pro": "gemini-1.5-pro-latest" 
                }
                model_id = gemini_model_mapping.get(selected_model, "gemini-1.5-pro-latest") 

                url = f'https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GOOGLE_API_KEY}'
                headers = {'Content-Type': 'application/json'}
                data = {"contents": [{"parts": [{"text": m["content"]}]}] for m in api_messages}

                response = requests.post(url, headers=headers, data=json.dumps(data))

                if response.status_code == 200:
                    response_json = response.json()
                    generated_text = response_json['candidates'][0]['content']['parts'][0]['text']
                else:
                    await message.channel.send(f"Error: {response.status_code}\n{response.text}")
                    return
            else:
                chat_completion = client.chat.completions.create(
                    messages=api_messages,
                    model=selected_model
                )
                generated_text = chat_completion.choices[0].message.content
                lobotomised_generated_text = generated_text[:2000] 
            await message.channel.send(lobotomised_generated_text.strip())

            logging.info(f"User: {message.author} - Message: {message.content} - Generated Text: {generated_text}")
            print(f"user:{message.author}\n message:{message.content}\n output:{generated_text}")

            messages.append({"role": "user", "content": message.content})
            messages.append({"role": "assistant", "content": generated_text.strip()})
            conversation_data[channel_id]["messages"] = messages[-10:] 

        except Exception as e:
            await message.channel.send(f"An error occurred: {e}")
            print(e)

# --- Event Handling ---

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    logging.info(f'Logged in as {bot.user.name}')
    await bot.tree.sync(guild=None)
    print("Application commands synced.")
    print("Connected to the following guilds:")
    logging.info("Application commands synced.")
    logging.info("Connected to the following guilds:")
    for guild in bot.guilds:
        print(f"  - {guild.name} (ID: {guild.id})")
        logging.info(f"  - {guild.name} (ID: {guild.id})")

      


# Run the bot
bot.run(DISCORD_TOKEN)
