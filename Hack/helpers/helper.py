import requests

def get_scraped_text(url):

    import requests

    jina_url = 'https://r.jina.ai/'+url
    print("*********************************",jina_url)
    headers = {
        'X-No-Cache': 'true',
        'X-Remove-Selector': 'footer, nav'
    }

    response = requests.get(jina_url, headers=headers)

    print(response.text)
    return response.text


