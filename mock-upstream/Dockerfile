# Мок-апстрим: статический бинарь, минимальный образ.
FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY *.go ./
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/mockup .

FROM gcr.io/distroless/static-debian12
COPY --from=build /out/mockup /mockup
EXPOSE 9000
ENTRYPOINT ["/mockup"]
CMD ["-addr", ":9000"]
