#!/usr/bin/env python3

import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from groq import Groq
from collections import defaultdict
import requests
import json
from datetime import datetime, timedelta
import google.generativeai as gemini
import asyncio
# Load environment variables from .env file
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') 
WHISPER_CPP_PATH = os.getenv('WHISPER_CPP_PATH')  # Path to the whisper.cpp executable
WHISPER_MODEL = os.getenv('WHISPER_CPP_MODEL')   # Specify the Whisper model you want to use
# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Initialize the Google Generative AI client
gemini.configure(api_key=GOOGLE_API_KEY)

# Initialize the bot with intents
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Authorized users/roles (replace with actual IDs)
authorized_users = [936673139419664414]   # Replace with user IDs
authorized_roles = [1198707036070871102]   # Replace with role IDs

# Bot-wide settings
bot_settings = {
    "model": "llama3-70b-8192",
    "system_prompt": "You are a helpful and friendly AI assistant.",
    "context_messages": 5,
    "llm_enabled": False  # LLM is enabled by default for the entire bot
}

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
    "gemini-1.5-pro-latest" 
]


    # --- Conversation Data (Important!) chatting shit
conversation_data = defaultdict(lambda: {"messages": []}) 


# --- Helper Function for Checking Authorization ---

def is_authorized(interaction: discord.Interaction):
    """Check if the user has permission to use the command."""
    user = interaction.user
    if user.id in authorized_users:
        return True
    if any(role.id in authorized_roles for role in user.roles):
        return True
    if user.id == interaction.guild.owner_id:
        return True
    return False
# --- Application Commands ---

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
        # Search for repositories
        url = "https://api.github.com/search/repositories"
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": 1 # Get top 5 matching repos
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        matching_repos = response.json()["items"]

        if matching_repos:
            embed = discord.Embed(
                title=f"GitHub Project Search Results for: {query}",
                color=discord.Color.green()  # Use a different color for search
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
        # Get trending repositories
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

                # Create the field value with the link SEPARATELY:
                field_value = f"{description}\n"
                field_value += f"⭐ {repo['stargazers_count']}   "
                field_value += f"🍴 {repo['forks_count']}"

                # Add the field with the name as the link:
                embed.add_field(
                        name=f"{repo_name}",  # Only the repo name here, no bolding or linking 
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


@bot.tree.command(name="summarize", description="Summarize a text using the current LLM.")
async def summarize(interaction: discord.Interaction, text: str):

    try:
        selected_model = bot_settings["model"]

        if selected_model in gemini_models:
            try:
                # Create a Gemini model instance (do this once, maybe outside the function)
                gemini_model = gemini.GenerativeModel(selected_model) 

                # Use the model instance to generate content
                response = gemini_model.generate_content( 
                    f"Summarize the following text:\n\n{text}",
                )

                # Extract the summary from the response
                summary = response.text
                await interaction.response.send_message(f"Summary:\n```\n{summary}\n```")
            except Exception as e:
                await interaction.response.send_message(f"An error occurred while processing the request: {e}")

        else: # Use Groq API for summarization
            system_prompt = bot_settings["system_prompt"]
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
                ],
                model=selected_model
            )
            summary = chat_completion.choices[0].message.content
            await interaction.response.send_message(f"Summary:\n```\n{summary}\n```")

    except Exception as e:
        await interaction.response.send_message(f"An error occurred: {e}")


@bot.tree.command(name="play_audio", description="Join a voice channel and play audio. (Authorized users only)")
async def play_audio(interaction: discord.Interaction, channel: discord.VoiceChannel):
    """Joins a specified voice channel and plays an audio file.

    Args:
        channel: The voice channel to join.
    """

    if not is_authorized(interaction):
        await interaction.response.send_message(
            "You are not authorized to use this command.", ephemeral=True
        )
        return

    # Check if the bot is already connected to a voice channel
    if interaction.guild.voice_client:
        await interaction.response.send_message(
            "The bot is already connected to a voice channel.", ephemeral=True
        )
        return

    try:
        # Connect to the specified voice channel
        await channel.connect()
        await interaction.response.send_message(
            f"Connected to voice channel: {channel.name}", ephemeral=True
        )

        # Path to your audio file (replace with the actual path)
        audio_file = r"path to audio shit"

        # Create a StreamPlayer for the audio
        source = discord.FFmpegPCMAudio(audio_file)
        interaction.guild.voice_client.play(source)

        # Wait until the audio finishes playing
        while interaction.guild.voice_client.is_playing():
            await asyncio.sleep(1)

        # Disconnect from the voice channel
        await interaction.guild.voice_client.disconnect()

    except Exception as e:
        await interaction.response.send_message(
            f"An error occurred: {e}", ephemeral=True
        )


@bot.tree.command(name="set_context_messages", description="Set the number of context messages to use (1-10) for the entire bot.")
async def set_context_messages(interaction: discord.Interaction, num_messages: int):
    if not is_authorized(interaction):
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    if 1 <= num_messages <= 10:
        bot_settings["context_messages"] = num_messages
        await interaction.response.send_message(f"Number of context messages set to: {num_messages} for the entire bot.")
    else:
        await interaction.response.send_message("Invalid number of messages. Please choose between 1 and 10.")

@bot.tree.command(name="say", description="Make the bot say something.")
async def say(interaction: discord.Interaction, message: str):
    if not is_authorized(interaction):
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    # Delete the user's command invocation (it will briefly appear)
    await interaction.response.defer(ephemeral=True) 
    await interaction.delete_original_response()

    # Send the message as the bot
    await interaction.channel.send(message)


@bot.tree.command(name="toggle_llm", description="Turn the LLM part of the bot on or off for the entire bot.")
async def toggle_llm(interaction: discord.Interaction):
    if not is_authorized(interaction):
        await interaction.response.send_message("You are not authorized to use this command.", ephemeral=True)
        return

    bot_settings["llm_enabled"] = not bot_settings["llm_enabled"]
    new_state = "OFF" if not bot_settings["llm_enabled"] else "ON"
    await interaction.response.send_message(f"LLM is now turned {new_state} for the entire bot.")

# --- Message Handling --- 

@bot.event
async def on_message(message):
    await bot.process_commands(message)

    if message.author == bot.user or not bot_settings["llm_enabled"]:
        return

    is_mentioned = bot.user.mentioned_in(message)
    is_reply_to_bot = message.reference is not None and message.reference.resolved.author == bot.user

    if is_mentioned or is_reply_to_bot:
        try:
            channel_id = str(message.channel.id)
            messages = conversation_data[channel_id]["messages"]  
            selected_model = bot_settings["model"]              
            system_prompt = bot_settings["system_prompt"]      
            context_messages_num = bot_settings["context_messages"] 

            context_messages = messages[-context_messages_num:]
            api_messages = [{"role": "system", "content": system_prompt}] + context_messages + [{"role": "user", "content": message.content}]

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
            else: # Groq model
                chat_completion = client.chat.completions.create(
                    messages=api_messages,
                    model=selected_model
                )
                generated_text = chat_completion.choices[0].message.content

            await message.channel.send(generated_text.strip())

            messages.append({"role": "user", "content": message.content})
            messages.append({"role": "assistant", "content": generated_text.strip()})
            conversation_data[channel_id]["messages"] = messages[-10:] 

        except Exception as e:
            await message.channel.send(f"An error occurred: {e}")




# --- Event Handling ---

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    await bot.tree.sync(guild=None)
    print("Application commands synced.")

# Run the bot
bot.run(DISCORD_TOKEN)