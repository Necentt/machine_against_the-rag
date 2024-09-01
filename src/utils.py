import requests
import uuid
import json
import datetime
import ssl
import aiohttp
import asyncio
import numpy as np
import random
import os
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from ast import literal_eval

load_dotenv()
giga_cred = os.getenv('GIGA_CRED')

def seed_all(seed=0):
    np.random.seed(seed)
    random.seed(seed)


def reacquire_giga(env_var="GIGA_AUTH"):
    return get_giga_auth(os.environ[env_var], verbose=False, team=True)


def get_giga_auth(GIGA_AUTH=giga_cred, verbose=False, team=False):
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    rquid = str(uuid.uuid4())
    payload = "scope=GIGACHAT_API_CORP" if team else "scope=GIGACHAT_API_PERS"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": rquid,
        "Authorization": "Basic " + GIGA_AUTH,
    }

    response = requests.request(
        "POST",
        url,
        headers=headers,
        data=payload,
        verify=False,
    )

    auth_token_data = literal_eval(response.text)
    access_token, expires_at = (
        auth_token_data["access_token"],
        datetime.datetime.fromtimestamp(auth_token_data["expires_at"] / 1000).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    )
    if verbose:
        print(f"GIGA creds expire at {expires_at}")
    return access_token


def get_giga_models(token):
    url = "https://gigachat.devices.sberbank.ru/api/v1/models"

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }

    response = requests.request("GET", url, headers=headers, verify=False)

    return response.json()


@retry(wait=wait_random_exponential(min=3, max=5), stop=stop_after_attempt(6))
def giga_send(
        phrase,
        token,
        profanity_check=True,
        model='GigaChat-Pro-preview',
        url="https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
        temperature=0.7,
        top_p=0.6,
        n=1,
        system_prompt: str = '',
        max_tokens=512,
):
    if system_prompt == "":
        msgs = [
            {"role": "user", "content": phrase}
        ]
    else:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": phrase}
        ]
    payload = json.dumps(
        {
            "model": model,
            "messages": msgs,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": False,
            "profanity_check": profanity_check,
            "max_tokens": max_tokens,
            "repetition_penalty": 1,
            "update_interval": 0,
        }
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    response = requests.request(
        "POST",
        url,
        headers=headers,
        data=payload,
        verify=False,
    )
    if response.status_code != 200:
        print(response.status_code)
        print(response.json())
    return response


@retry(wait=wait_random_exponential(min=3, max=5), stop=stop_after_attempt(6))
async def giga_send_async(
        phrase,
        token,
        session,
        semaphore,
        profanity_check=True,
        model='GigaChat-Pro-preview',
        url="https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
        temperature=0.7,
        top_p=0.6,
        n=1,
        system_prompt: str = '',
        max_tokens=512,
):
    if system_prompt == "":
        msgs = [
            {"role": "user", "content": phrase}
        ]
    else:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": phrase}
        ]
    payload = {
        "model": model,
        "messages": msgs,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stream": False,
        "profanity_check": profanity_check,
        "max_tokens": max_tokens,
        "repetition_penalty": 1,
        "update_interval": 0,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    async with semaphore:
        async with session.post(url, json=payload, headers=headers) as response:
            # print(payload, headers)
            return await response.json()


async def giga_send_async_bulk(
        sys_prompt,
        prompts,
        giga_tok,
        semaphore=10,
        max_tokens=512,
        top_p=0.6,
        temperature=0.7,
        profanity_check=True,
        n=1,
        model='GigaChat-Pro-preview',
):
    ssl_context = ssl.create_default_context(cafile='/etc/ssl/certs/ca-certificates.crt')

    semaphore = asyncio.Semaphore(semaphore)
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            giga_send_async(
                phrase=prompt,
                token=giga_tok,
                session=session,
                semaphore=semaphore,
                system_prompt=sys_prompt,
                profanity_check=profanity_check,
                model=model,
                url="https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
                temperature=temperature,
                top_p=top_p,
                n=n,
                max_tokens=max_tokens,
            ) for prompt in prompts
        ]
        results = await asyncio.gather(*tasks)
        return results


def giga_make_embeddings(
        texts,
        giga_tok
) -> np.array:
    url = "https://gigachat.devices.sberbank.ru/api/v1/embeddings"

    payload = json.dumps({
        "model": "Embeddings",
        "input": texts
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {giga_tok}'
    }

    response = requests.request(
        "POST",
        url,
        headers=headers,
        data=payload,
        verify=False,
    )
    print(response.json())
    return np.array([i['embedding'] for i in response.json()['data']])
