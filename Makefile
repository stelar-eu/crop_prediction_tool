
DOCKER=docker
IMGTAG=img.stelar.gr/stelar/crop-classification:latest
IMGPATH=.
DOCKERFILE=$(IMGPATH)/Dockerfile.ovwrt

.PHONY: all build push


all: build push

build:
	$(DOCKER) build -f $(DOCKERFILE) $(IMGPATH) -t $(IMGTAG)

push:
	$(DOCKER) push $(IMGTAG)
