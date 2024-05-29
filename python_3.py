import glob
import io
import os
import shutil
import sys
import time

import numpy as np
import torch
from ns import ns
from ns.applications import Application
from ns.core import TypeId
from ns.network import InetSocketAddress, Ipv4Address, Packet, Socket
from ns.simulator import Simulator
from omegaconf import OmegaConf
from PIL import Image

from model.net import AE

PORT_NUMBER = 9
droot = "/home/swjung/ns-allinone-3.41/ns-3.41/scratch/cifar10"

# Creating Nodes
nodes = ns.network.NodeContainer()
nodes.Create(3)

# Creating P2P links
p2p = ns.point_to_point.PointToPointHelper()
p2p.SetDeviceAttribute("DataRate", ns.core.StringValue("30Mbps"))
p2p.SetChannelAttribute("Delay", ns.core.StringValue("2ms"))

# Connecting Nodes in a row using P2P links
device1 = p2p.Install(nodes.Get(0), nodes.Get(1))
device2 = p2p.Install(nodes.Get(1), nodes.Get(2))

# Assigning Addresses
stack = ns.internet.InternetStackHelper()
stack.Install(nodes)

address1 = ns.internet.Ipv4AddressHelper()
address1.SetBase(
    ns.network.Ipv4Address("10.1.1.0"), ns.network.Ipv4Mask("255.255.255.0")
)
interface1 = address1.Assign(device1)

address2 = ns.internet.Ipv4AddressHelper()
address2.SetBase(
    ns.network.Ipv4Address("10.2.2.0"), ns.network.Ipv4Mask("255.255.255.0")
)
interface2 = address2.Assign(device2)

ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

# Define callback function and event implementer using cppyy
ns.cppyy.cppdef(
    """
    namespace ns3
    {
        Callback<void, Ptr<Socket>> make_rx_callback(void(*f)(Ptr<Socket>))
        {
            return MakeCallback(f);
        }
        EventImpl* GeneratorEventSend(void(*f)(Ptr<Socket>, InetSocketAddress&, int), Ptr<Socket> socket, InetSocketAddress address, int index)
        {
            return MakeEvent(f, socket, address, index);
        }
        EventImpl* ForwarderEventSend(void(*f)(Ptr<Socket>, Ptr<Packet>), Ptr<Socket> socket, Ptr<Packet> packet)
        {
            return MakeEvent(f, socket, packet);
        }
    }
"""
)


class Generator(ns.applications.Application):
    socketToInstanceDict = {}

    # Initialization
    def __init__(self, node: ns.Node, send_address: ns.Ipv4Address):
        super().__init__()
        self.__python_owns__ = False  # let C++ destroy this on Simulator::Destroy
        self.port = PORT_NUMBER
        self.send_address = send_address
        self.send_socket = ns.network.Socket.CreateSocket(
            node, ns.core.TypeId.LookupByName("ns3::UdpSocketFactory")
        )
        Generator.socketToInstanceDict[self.send_socket] = self
        self.m_sent = 0

    def __del__(self):
        del Generator.socketToInstanceDict[self.send_socket]

    def StartApplication(self):
        self.send_socket.Connect(
            ns.network.InetSocketAddress(self.send_address, self.port).ConvertTo()
        )
        self.ScheduleTransmit(ns.core.Seconds(0.0))

    def StopApplication(self):
        ns.Simulator.Cancel(self.m_sendEvent)
        self.send_socket.Close()

    def ScheduleTransmit(self, delay):
        address = ns.network.InetSocketAddress(self.send_address, self.port)
        # XXX Just send 3 images in this example
        for index in range(3):
            event = ns.GeneratorEventSend(
                Generator._Send, self.send_socket, address, index
            )
            # Schedule the event without delay
            self.m_sendEvent = ns.Simulator.Schedule(ns.Seconds(0.001 * index), event)

    def Send(self, address: ns.InetSocketAddress, index) -> None:
        img_path = os.listdir(droot)[index]
        img_byte_array = io.BytesIO()

        # Load raw image (dtype: uint8)
        image = Image.open(os.path.join(droot, img_path))

        # write image file to buffer using numpy.
        # PIL image has a shape of (H, W, 3)
        # -> should be permuted to (3, H, W) for neural network computation.
        np.save(img_byte_array, np.array(image, dtype=np.uint8).transpose(2, 0, 1))

        image_bytes = img_byte_array.getvalue()
        image_buffer = memoryview(image_bytes)

        # Packets to contain images
        packet = ns.network.Packet(image_buffer, len(image_buffer))
        self.send_socket.Send(packet)
        self.m_sent += 1
        print(
            "At time +{s}s Generator sent {i}th packet of {b} bytes from {ip} port {port}".format(
                s=ns.Simulator.Now().GetSeconds(),
                i=self.m_sent,
                b=packet.GetSize(),
                ip=address.GetIpv4(),
                port=address.GetPort(),
            ),
            file=sys.stderr,
            flush=True,
        )

    def _Send(socket: ns.Socket, address: ns.InetSocketAddress, index):
        instance = Generator.socketToInstanceDict[socket]
        instance.Send(address, index)


class Forwarder(ns.applications.Application):
    socketToInstanceDict = {}

    def __init__(
        self, node: ns.Node, rx_address: ns.Ipv4Address, tx_address: ns.Ipv4Address
    ):
        super().__init__()
        self.__python_owns__ = False  # let C++ destroy this on Simulator::Destroy
        self.port = PORT_NUMBER  # 'Listen port' for the server
        self.rx_address = rx_address
        self.tx_address = tx_address

        # UDP socket for receiving
        self.receive_socket = ns.network.Socket.CreateSocket(
            node, ns.core.TypeId.LookupByName("ns3::UdpSocketFactory")
        )
        self.receive_socket.Bind(
            ns.network.InetSocketAddress(self.rx_address, self.port).ConvertTo()
        )

        # UDP socket for forwarding
        self.send_socket = ns.network.Socket.CreateSocket(
            node, ns.core.TypeId.LookupByName("ns3::UdpSocketFactory")
        )

        # Callback: Make '_Receive' be called upon a packet reception
        self.receive_socket.SetRecvCallback(self._Receive)
        Forwarder.socketToInstanceDict[self.receive_socket] = self
        Forwarder.socketToInstanceDict[self.send_socket] = self

        # Load pre-trained neural networks.
        self.nets = []
        for lv in range(3):
            cfg = OmegaConf.create(
                {
                    "lv": lv + 1,
                    "in_ch": 3 * (2**lv),
                    "out_ch": 3 * (2 ** (lv + 1)),
                    "f_size": 3,
                    "lr": 0.0002,
                }
            )
            self.nets.append(AE(cfg))
        for lv, net in enumerate(self.nets):
            net.load_state_dict(torch.load("myScratch/model/Lv{}.pth".format(lv)))
            net.eval()

    def __del__(self):
        del Forwarder.socketToInstanceDict[self.receive_socket]
        del Forwarder.socketToInstanceDict[self.send_socket]

    def Send(self, packet: ns.Packet) -> None:
        # Get the payload from the received packet
        original_data = bytearray(packet.GetSize())
        packet.CopyData(original_data, packet.GetSize())

        ######################################
        ######### NN Computation #############
        ######################################
        # Compression with neural network.
        # - Original data is (3, H, W)
        #   -> flattened to (3 * H * W,)
        #   -> reshaped to (3, H, W)
        if 3:
            x = (
                torch.tensor(
                    np.frombuffer(original_data, dtype=np.uint8)[128:],
                    dtype=torch.float32,
                ).reshape(3, 32, 32)
                / 255
            )
            for lv in range(3):
                x = self.nets[lv].encode(x)  # encode data
            x = x.detach().cpu().numpy()
            buff = io.BytesIO()
            np.save(buff, x)
            new_data = buff.getvalue()
        else:
            new_data = original_data

        modified_packet = Packet(new_data)
        self.send_socket.SendTo(
            modified_packet, 0, InetSocketAddress(self.tx_address, self.port)
        )

        print(
            "At time +{s}s Forwarder sent a packet of {b} bytes from {ip} port {port}".format(
                s=ns.Simulator.Now().GetSeconds(),
                b=modified_packet.GetSize(),
                ip=self.tx_address,
                port=self.port,
            ),
            file=sys.stderr,
            flush=True,
        )

    def Receive(self):
        address = ns.Address()
        packet = self.receive_socket.RecvFrom(address)
        inetAddress = ns.InetSocketAddress.ConvertFrom(address)

        print(
            "At time +{s}s Forwarder received {b} bytes from {ip} port {port}".format(
                s=ns.Simulator.Now().GetSeconds(),
                b=packet.GetSize(),
                ip=inetAddress.GetIpv4(),
                port=inetAddress.GetPort(),
            ),
            file=sys.stderr,
            flush=True,
        )

        # Schedule the event to forward the packet
        event = ns.ForwarderEventSend(Forwarder._Send, self.send_socket, packet)
        ns.Simulator.Schedule(ns.Seconds(0), event)

    @staticmethod
    def _Send(socket: ns.Socket, packet: ns.Packet):
        instance = Forwarder.socketToInstanceDict[socket]
        instance.Send(packet)

    @staticmethod
    def _Receive(socket: ns.Socket) -> None:
        instance = Forwarder.socketToInstanceDict[socket]
        instance.Receive()


class Receiver(ns.applications.Application):
    socketToInstanceDict = {}

    def __init__(self, node: ns.Node, address: ns.Ipv4Address):
        super().__init__()
        self.__python_owns__ = False  # Let C++ destroy this on Simulator::Destroy
        self.port = PORT_NUMBER
        self.rx_address = address  # Address of the source node (Node 0)

        # Listening UDP socket for receiving -- receive from any
        self.receive_socket = ns.network.Socket.CreateSocket(
            node, ns.core.TypeId.LookupByName("ns3::UdpSocketFactory")
        )
        self.receive_socket.Bind(
            ns.network.InetSocketAddress(
                ns.network.Ipv4Address.GetAny(), self.port
            ).ConvertTo()
        )

        # Callback -- make _Receive be called upon a pkt reception
        self.receive_socket.SetRecvCallback(ns.make_rx_callback(Receiver._Receive))
        Receiver.socketToInstanceDict[self.receive_socket] = self
        self.m_received = 0

        # Load pre-trained neural networks.
        self.nets = []
        for lv in range(3):
            cfg = OmegaConf.create(
                {
                    "lv": lv + 1,
                    "in_ch": 3 * (2**lv),
                    "out_ch": 3 * (2 ** (lv + 1)),
                    "f_size": 3,
                    "lr": 0.0002,
                }
            )
            self.nets.append(AE(cfg))
        for lv, net in enumerate(self.nets):
            net.load_state_dict(torch.load("myScratch/model/Lv{}.pth".format(lv)))
            net.eval()

        # Shapes of encoded latents
        self.lv_shapes = {0: (3, 32, 32), 1: (6, 8, 8), 2: (12, 4, 4), 3: (24, 2, 2)}

        # Packet arrival times
        self.tArrivals = []

        # Directory to store the received images
        # If the dir exist, delelte the previous one and re-make
        self.imsave_dir = "./arrivals/dest/"
        if os.path.exists(self.imsave_dir):
            shutil.rmtree(self.imsave_dir)
        os.makedirs(self.imsave_dir, exist_ok=True)

    def __del__(self):
        del Receiver.socketToInstanceDict[self.receive_socket]

    def Receive(self):
        address = ns.Address()
        packet = self.receive_socket.RecvFrom(address)
        inetAddress = ns.InetSocketAddress.ConvertFrom(address)

        # Get the payload from the received packet
        buf = bytearray(packet.GetSize())
        packet.CopyData(buf, packet.GetSize())
        image_bytes = bytes(buf)

        # Restore the image file from the received packet
        if 3:
            x = torch.tensor(
                np.frombuffer(image_bytes, dtype=np.float32)[32:], dtype=torch.float32
            ).reshape(self.lv_shapes[3])
            # Decode latent
            for net in reversed(self.nets[:3]):
                x = net.decode(x)
            # Latent to image
            x = (x.permute(1, 2, 0).detach().cpu() * 255).to(torch.uint8).numpy()
            img = Image.fromarray(x)
        else:
            img = Image.fromarray(
                np.frombuffer(image_bytes, dtype=np.uint8)[128:]
                .reshape(self.lv_shapes[3])
                .transpose(1, 2, 0)
            )

        # Store the image in the directory
        image_filename = os.path.join(
            self.imsave_dir, "Arrival_{}.png".format(self.m_received)
        )
        img.save(image_filename)

        print(
            "At time +{s}s Receiver received {b} bytes from {ip} port {port}".format(
                s=ns.Simulator.Now().GetSeconds(),
                b=packet.__deref__().GetSize(),
                ip=inetAddress.GetIpv4(),
                port=inetAddress.GetPort(),
            ),
            file=sys.stderr,
            flush=True,
        )
        print(
            "At time +{}s Receiver stored the image {}".format(
                ns.Simulator.Now().GetSeconds(), self.m_received
            ),
            file=sys.stderr,
            flush=True,
        )

    @staticmethod
    def _Receive(socket: ns.Socket) -> None:
        instance = Receiver.socketToInstanceDict[socket]
        instance.Receive()


# Initialize applications and install at appropriate nodes
generatorApp = Generator(nodes.Get(0), interface1.GetAddress(1))
nodes.Get(0).AddApplication(generatorApp)

forwarderApp = Forwarder(
    nodes.Get(1), interface1.GetAddress(0), interface2.GetAddress(1)
)
nodes.Get(1).AddApplication(forwarderApp)

receiverApp = Receiver(nodes.Get(2), interface1.GetAddress(0))
nodes.Get(2).AddApplication(receiverApp)

# Set start&stop time of applications
AppContainer = ns.ApplicationContainer()
AppContainer.Add(generatorApp)
AppContainer.Add(forwarderApp)
AppContainer.Add(receiverApp)
AppContainer.Start(ns.core.Seconds(1.0))
AppContainer.Stop(ns.core.Seconds(10.0))

ns.core.Simulator.Run()
ns.core.Simulator.Destroy()
