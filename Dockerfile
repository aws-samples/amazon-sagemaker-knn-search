#Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#SPDX-License-Identifier: MIT-0
#Base docker image
FROM amazonlinux:2

RUN yum install python3 -y

#Installing python packages from requirements file
COPY ./src/requirements.txt /opt/app/requirements.txt
RUN pip3 install -r /opt/app/requirements.txt
RUN echo "Done installing requirements"

ENV PYTHONUNBUFFERED=TRUE
ENTRYPOINT ["python3"]
