FROM python:3.6

# set the working directory
# RUN ["mkdir", "app"]
   
WORKDIR /opt/program

ADD config /opt/program

RUN ["pip", "install", "-r", "requirements.txt"]

# install environment dependencies
ADD input /opt/program/input
ADD model /opt/program/model
ADD output /opt/program/output


# provision environment
RUN ["chmod", "+x", "./run.sh"]

EXPOSE 8080
ENTRYPOINT ["./run.sh"]

ENV PATH="/opt/program:${PATH}"


