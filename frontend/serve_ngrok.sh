#!/bin/bash
while ! docker run -it -e NGROK_AUTHTOKEN="${!NGROK_AUTHTOKEN}" ngrok/ngrok http 11700 --subdomain=captafied; do    sleep 1
done