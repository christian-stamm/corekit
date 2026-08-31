cmake_minimum_required(VERSION 3.12)

set(WIZNET_CHIP 6300)
set(WIZNET_DIR $ENV{WIZNET_DRIVER_PATH})

add_compile_options(
    -Wno-format          
    -Wno-unused-variable
    -Wno-return-type
    -Wno-pointer-sign
)

# Ethernet
add_library(ETHERNET_FILES INTERFACE)

target_sources(ETHERNET_FILES INTERFACE
    ${WIZNET_DIR}/Ethernet/socket.c
    ${WIZNET_DIR}/Ethernet/wizchip_conf.c
    ${WIZNET_DIR}/Ethernet/W${WIZNET_CHIP}/w${WIZNET_CHIP}.c
)

target_include_directories(ETHERNET_FILES INTERFACE
        ${WIZNET_DIR}
        ${WIZNET_DIR}/Ethernet
        ${WIZNET_DIR}/Ethernet/W${WIZNET_CHIP}
)

target_compile_definitions(ETHERNET_FILES INTERFACE
        -D_WIZCHIP_=W${WIZNET_CHIP}
        -D_WIZCHIP_QSPI_MODE_=QSPI_QUAD_MODE
)

add_library(wiznet::w${WIZNET_CHIP} ALIAS ETHERNET_FILES)

# Loopback
add_library(LOOPBACK_FILES STATIC
        ${WIZNET_DIR}/Application/loopback/loopback.c
)

target_include_directories(LOOPBACK_FILES PUBLIC
        ${WIZNET_DIR}/Application/loopback
)

target_link_libraries(LOOPBACK_FILES PUBLIC
        ETHERNET_FILES
)


# DHCP
add_library(AAC_FILES STATIC
        ${WIZNET_DIR}/Internet/AAC/AddressAutoConfig.c
)

target_include_directories(AAC_FILES PUBLIC

        ${WIZNET_DIR}/Internet/AAC
)

target_link_libraries(AAC_FILES PUBLIC
        ETHERNET_FILES
)

# DHCP
add_library(DHCP_FILES STATIC
        ${WIZNET_DIR}/Internet/DHCP/dhcp.c
)

target_include_directories(DHCP_FILES PUBLIC
        ${WIZNET_DIR}/Internet/DHCP
)

target_link_libraries(DHCP_FILES PUBLIC
        ETHERNET_FILES
)

# DHCP
add_library(DHCP6_FILES STATIC
        ${WIZNET_DIR}/Internet/DHCP6/dhcpv6.c
)

target_include_directories(DHCP6_FILES PUBLIC
        ${WIZNET_DIR}/Internet/DHCP6
)

target_link_libraries(DHCP6_FILES PUBLIC
        ETHERNET_FILES
)

# DNS
add_library(DNS_FILES STATIC
        ${WIZNET_DIR}/Internet/DNS/dns.c
)

target_include_directories(DNS_FILES PUBLIC
        ${WIZNET_DIR}/Internet/DNS
)

target_link_libraries(DNS_FILES PUBLIC
        ETHERNET_FILES
)

# HTTP Server
add_library(HTTPSERVER_FILES STATIC
        ${WIZNET_DIR}/Internet/httpServer/httpParser.c
        ${WIZNET_DIR}/Internet/httpServer/httpServer.c
        ${WIZNET_DIR}/Internet/httpServer/httpUtil.c
)

target_include_directories(HTTPSERVER_FILES PUBLIC
        ${WIZNET_DIR}/Internet/httpServer
)

target_link_libraries(HTTPSERVER_FILES PUBLIC
        ETHERNET_FILES
)

# MQTT
add_library(MQTT_FILES STATIC
        ${WIZNET_DIR}/Internet/MQTT/mqtt_interface.c
        ${WIZNET_DIR}/Internet/MQTT/MQTTClient.c
        ${WIZNET_DIR}/Internet/MQTT/MQTTPacket/src/MQTTConnectClient.c
        ${WIZNET_DIR}/Internet/MQTT/MQTTPacket/src/MQTTConnectServer.c
        ${WIZNET_DIR}/Internet/MQTT/MQTTPacket/src/MQTTDeserializePublish.c
        ${WIZNET_DIR}/Internet/MQTT/MQTTPacket/src/MQTTFormat.c
        ${WIZNET_DIR}/Internet/MQTT/MQTTPacket/src/MQTTPacket.c
        ${WIZNET_DIR}/Internet/MQTT/MQTTPacket/src/MQTTSerializePublish.c
        ${WIZNET_DIR}/Internet/MQTT/MQTTPacket/src/MQTTSubscribeClient.c
        ${WIZNET_DIR}/Internet/MQTT/MQTTPacket/src/MQTTSubscribeServer.c
        ${WIZNET_DIR}/Internet/MQTT/MQTTPacket/src/MQTTUnsubscribeClient.c
        ${WIZNET_DIR}/Internet/MQTT/MQTTPacket/src/MQTTUnsubscribeServer.c
)

target_include_directories(MQTT_FILES 
PUBLIC
        ${WIZNET_DIR}/Internet/MQTT
PRIVATE
        ${WIZNET_DIR}/Internet/MQTTPacket/src
)

target_link_libraries(MQTT_FILES PUBLIC
        ETHERNET_FILES
)

# SNTP
add_library(SNTP_FILES STATIC
        ${WIZNET_DIR}/Internet/SNTP/sntp.c
)

target_include_directories(SNTP_FILES PUBLIC
        ${WIZNET_DIR}/Internet/SNTP
)

target_link_libraries(SNTP_FILES PUBLIC
        ETHERNET_FILES
)
        
# SNTP
add_library(SNMP_FILES STATIC
        ${WIZNET_DIR}/Internet/SNMP/snmp.c
)

target_include_directories(SNMP_FILES PUBLIC
        ${WIZNET_DIR}/Internet/SNMP
)

target_link_libraries(SNMP_FILES PUBLIC
        ETHERNET_FILES
)
        
# TFTP Server
add_library(TFTP_FILES STATIC
        ${WIZNET_DIR}/Internet/TFTP/tftp.c
        ${WIZNET_DIR}/Internet/TFTP/netutil.c
)

target_include_directories(TFTP_FILES PUBLIC
        ${WIZNET_DIR}/Internet/TFTP
)

target_link_libraries(TFTP_FILES PUBLIC
        ETHERNET_FILES
)

# multicast
add_library(MULTI_FILES STATIC
        ${WIZNET_DIR}/Application/multicast/multicast.c
)

target_include_directories(MULTI_FILES PUBLIC
        ${WIZNET_DIR}/Application/multicast
)

target_link_libraries(MULTI_FILES PUBLIC
        ETHERNET_FILES
)