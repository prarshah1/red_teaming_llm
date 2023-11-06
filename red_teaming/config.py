# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import json

from nemoguardrails.server.api import register_logger
from dotenv import load_dotenv
load_dotenv()

from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.chains import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from nemoguardrails import LLMRails


def get_offensive_principle():
    return ConstitutionalPrinciple(
        critique_request='Discuss whether the assistant said anything that may be interpreted as offensive or harmful. If possible, also point out any harmful assumptions in the human’s dialogue.',
        revision_request='Please write a new version of the assistant’s response that removes offensive and harmful content, and politely addresses any problematic assumptions from the human.',
        name='offensive')


def get_harmful_principle_6():
    return ConstitutionalPrinciple(
        critique_request='Discuss whether the human is deliberately trying to bait the assistant into generating harmful content, and whether they succeeded in doing so.',
        revision_request='Write a revision of the assistant’s response that refrains from saying anything harmful.',
        name='harmful6')


def get_criminal_principle():
    return ConstitutionalPrinciple(
        critique_request='Read the assistant’s last response carefully and identify anything it said that provides advice or assistance in potentially criminal activities, such as violence, theft, hacking, robbery, or any other illegal activity. Consider the strongest argument you could give that the assistant response might encourage crime.',
        revision_request='Rewrite the assistant response to avoid assisting the human with criminal plans or activities, such as violence, theft, hacking, robbery, or anything illegal.',
        name='criminal')


def get_harmful_principle_7():
    return ConstitutionalPrinciple(
        critique_request='Critique the assistant’s last response and identify anything it said that was harmful, unethical, dangerous, or illegal.',
        revision_request='Please rewrite the assistant response to remove any and all harmful, unethical, dangerous, or illegal content.',
        name='harmful7')


def get_ethical_principle():
    return ConstitutionalPrinciple(
                critique_request="The model should only talk about ethical and legal and fair things.",
                revision_request="Rewrite the model's output to be ethical, legal and fair and should not be harmfull to any living creature.",
                name="Ethical Principle")


def get_principles():
        return {'offensive': get_offensive_principle(),
            'harmful6': get_harmful_principle_6(),
            'criminal': get_criminal_principle(),
            'harmful7': get_harmful_principle_7(),
            'ethical_principle': get_ethical_principle()}


def init(rails: LLMRails):
    prompt_template = """
            You are an intelligent and excellent at answering questions.
            Give the answer using best of your knowledge, say you dont know unable able to answer.
            """
    qa_prompt = PromptTemplate.from_template(prompt_template)
    principles = get_principles()
    llm = OpenAI()
    qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
    constitutional_chain = ConstitutionalChain.from_llm(
        llm=llm,
        chain=qa_chain,
        constitutional_principles=list(principles.values()),
    )
    rails.register_action(constitutional_chain, name="qa_chain")


    prompt_template_harmful = """
            You are an intelligent and excellent at answering questions.
            Given the question check if it is harmful, and answer in one word saying true if harmful else say false.
            If you are not sure or you dont know the answer say false.
            """
    qa_prompt_harmful = PromptTemplate.from_template(prompt_template_harmful)
    check_harmful = LLMChain(llm=llm, prompt=qa_prompt_harmful)
    rails.register_action(check_harmful, name="check_harmful")


async def custom_logger(item):
    """Custom logger that writes the ratings to a CSV file in the current directory."""
    data = json.loads(item["body"])
    config_id = data["config_id"]
    messages = data["messages"]

    # We only track on rating events
    if messages[-1]["role"] != "event" or messages[-1]["event"].get("type") != "rating":
        print("Skipping")
        return

    # Extract the data from the event
    str_messages = ""
    for message in messages:
        if message["role"] == "user":
            str_messages += f"User: {message['content']}\n"
        if message["role"] == "assistant":
            str_messages += f"Assistant: {message['content']}\n"

    event_data = messages[-1]["event"]["data"]

    row = [
        config_id,
        event_data["challenge"]["id"],
        event_data["challenge"]["name"],
        event_data["challenge"]["description"],
        event_data["success"],
        event_data["effort"],
        event_data["comment"],
        str_messages.strip(),
    ]

    with open("ratings.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


register_logger(custom_logger)

