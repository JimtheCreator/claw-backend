package main

import (
  "context"
  "fmt"
  "log"
  "net"
  "time"

  "google.golang.org/grpc"
  pb "path/to/generated/chartpb"
)

type server struct {
  pb.UnimplementedChartServiceServer
}

func (s *server) GetHistoricalChart(req *pb.ChartRequest, stream pb.ChartService_GetHistoricalChartServer) error {
  symbol := req.GetSymbol()
  from := time.Unix(req.GetFromTimestamp(), 0)
  to := time.Unix(req.GetToTimestamp(), 0)
  interval := req.GetInterval()

  // In production, this will query InfluxDB based on symbol/from/to/interval
  fmt.Printf("Fetching chart for %s from %v to %v at %s interval\n", symbol, from, to, interval)

  for i := 0; i < 100; i++ {
    point := &pb.ChartPoint{
      Timestamp: from.Unix() + int64(i*60),
      Open:      50000 + float64(i),
      High:      50010 + float64(i),
      Low:       49990 + float64(i),
      Close:     50005 + float64(i),
      Volume:    10.0 + float64(i),
    }
    if err := stream.Send(point); err != nil {
      return err
    }
  }

  return nil
}

func main() {
  lis, err := net.Listen("tcp", ":50051")
  if err != nil {
    log.Fatalf("failed to listen: %v", err)
  }

  grpcServer := grpc.NewServer()
  pb.RegisterChartServiceServer(grpcServer, &server{})

  log.Println("gRPC server running at :50051")
  if err := grpcServer.Serve(lis); err != nil {
    log.Fatalf("failed to serve: %v", err)
  }
}
