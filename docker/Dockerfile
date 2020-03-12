FROM rootproject/root:latest

# working directory
WORKDIR /sync

# environment variables
ENV PATH /sync/bin:$PATH
ENV PYTHONPATH /sync:$PYTHONPATH

# prepare yum
RUN yum -y update; yum clean all
RUN yum -y install yum-plugin-priorities yum-utils; yum clean all

# basic software
RUN yum -y install which nano less htop pandoc; yum clean all
RUN yum -y install krb5-libs krb5-server krb5-workstation; yum clean all

# configure kerberos
COPY krb5.conf /etc/krb5.conf

# python packages
RUN yum -y install python-pip; yum clean all
RUN pip install --upgrade pip
RUN pip install ipython
RUN pip install six
RUN pip install oyaml
RUN pip install numpy
RUN pip install scipy
RUN pip install matplotlib
RUN pip install pandas
RUN pip install numexpr
RUN pip install root_numpy
RUN pip install root_pandas
RUN pip install tabulate
RUN pip install networkx
