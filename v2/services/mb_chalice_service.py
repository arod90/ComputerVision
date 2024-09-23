import requests
import os
import boto3

class MBChaliceService:
    def __init__(self, api_key=None):
        self.base_url = "http://127.0.0.1:8000"
        self.api_key = api_key

    def _get_headers(self):
        headers = {
            'Content-Type': 'application/json',
        }
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers
    
    def post(self, data=None):
        url = f'{self.base_url}/usuario/image/test?is_ingreso=True'
        response = requests.post(url, headers=self._get_headers(), json=data)
        response.raise_for_status()
        return response.json()