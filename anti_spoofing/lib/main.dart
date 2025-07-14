import 'dart:async';
import 'dart:convert';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:socket_io_client/socket_io_client.dart' as IO;

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: 'Live Face Detection',
      debugShowCheckedModeBanner: false,
      home: FaceDetectionPage(),
    );
  }
}

class FaceDetectionPage extends StatefulWidget {
  const FaceDetectionPage({super.key});

  @override
  State<FaceDetectionPage> createState() => _FaceDetectionPageState();
}

class _FaceDetectionPageState extends State<FaceDetectionPage> {
  late CameraController _controller;
  late IO.Socket _socket;
  String message = "Waiting for detection...";
  double variance = 0.0;
  String? receivedImage;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _initCamera();
    _initSocket();
  }

  void _initCamera() async {
    _controller = CameraController(
      cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
    );
    await _controller.initialize();
    setState(() {});
    _startFrameSending();
  }

  void _initSocket() {
    _socket = IO.io(
      'https://steal-destroyed-subsidiaries-bobby.trycloudflare.com/',
      IO.OptionBuilder()
          .setTransports(['websocket'])
          .disableAutoConnect()
          .build(),
    );

    _socket.connect();

    _socket.onConnect((_) {
      print('Connected to server');
    });

    _socket.on('live_face_detected', (data) {
      setState(() {
        message = data['message'] ?? 'Live face detected';
        variance = data['variance'] ?? 0.0;
        receivedImage = data['image'];
      });

      // ✅ Stop sending more frames after live detection
      _timer?.cancel();
    });

    _socket.on('frame_result', (data) {
      setState(() {
        message = data['message'];
        variance = data['variance'] ?? 0.0;
        receivedImage = null;
      });
    });

    _socket.onDisconnect((_) {
      print('Disconnected from server');
    });
  }

  void _startFrameSending() {
    _timer = Timer.periodic(const Duration(seconds: 1), (_) async {
      if (!_controller.value.isInitialized) return;

      try {
        final xFile = await _controller.takePicture();
        final bytes = await xFile.readAsBytes();
        final base64Img = base64Encode(bytes);
        _socket.emit('video_frame', 'data:image/jpeg;base64,$base64Img');
      } catch (e) {
        print("Camera frame error: $e");
      }
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    _controller.dispose();
    _socket.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text("Face Liveness Detection")),
      body: SingleChildScrollView(
        child: Column(
          children: [
            AspectRatio(
              aspectRatio: _controller.value.aspectRatio,
              child: CameraPreview(_controller),
            ),
            const SizedBox(height: 10),
            Text("Message: $message"),
            Text("Variance: $variance"),
            const SizedBox(height: 10),
            if (receivedImage != null)
              Image.memory(
                base64Decode(receivedImage!.split(',')[1]),
                width: 300,
              )
          ],
        ),
      ),
    );
  }
}
