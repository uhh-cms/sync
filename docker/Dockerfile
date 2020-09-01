FROM rootproject/root:latest

# working directory
WORKDIR /sync

# environment variables
ENV PATH /sync/bin:$PATH
ENV PYTHONPATH /sync:$PYTHONPATH

# prepare apt
RUN apt -y update; apt clean
RUN apt -y install software-properties-common build-essential; apt clean
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt -y update; apt clean

# basic software
RUN apt -y install nano less htop pandoc; apt clean
RUN DEBIAN_FRONTEND=noninteractive apt -y install krb5-kdc krb5-admin-server krb5-config; apt clean

# configure kerberos
COPY krb5.conf /etc/krb5.conf

# setup python
RUN cd "$( dirname "$( which python3 )" )" && ln -s python3 python
RUN apt -y install python3-pip
RUN cd "$( dirname "$( which pip3 )" )" && ln -s pip3 pip

# python packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir ipython
RUN pip install --no-cache-dir six
RUN pip install --no-cache-dir oyaml
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir scipy
RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir pandas
RUN pip install --no-cache-dir numexpr
RUN pip install --no-cache-dir root_numpy
RUN pip install --no-cache-dir root_pandas
RUN pip install --no-cache-dir tabulate
RUN pip install --no-cache-dir networkx
