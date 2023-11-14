# For all files named *.sh in the directory run the script and write both stdout and stderr to a file ending in .log

SCRIPTS := $(wildcard *.sh)
LOGS := $(SCRIPTS:%.sh=%.log)

all: $(LOGS)

%.log: %.sh
	./$< 2>&1 | tee $@