/*
 * ============================================================
 * THEO ESP32-S3 HARDWARE NODE
 * ============================================================
 *
 * Hardware:
 *   - OV2640 camera
 *   - INMP441 I2S microphone
 *
 * Network API:
 *
 *   GET /
 *   GET /status
 *   GET /camera/capture
 *   GET /camera/settings
 *   GET /audio/record?duration=3000
 *
 * ============================================================
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <Preferences.h>

#include "esp_camera.h"
#include "ESP_I2S.h"
#include "esp_heap_caps.h"
#include "driver/gpio.h"


// ============================================================
// WIFI CONFIGURATION
// ============================================================

const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";


// Device hostname:
//
// http://theo-esp32.local
//
const char* HOSTNAME = "theo-esp32";


// ============================================================
// OV2640 PIN CONFIGURATION
// ============================================================
//
// Your camera:
//
// GND
// SCL
// SDA
// D0
// D2
// D4
// D6
// DCLK
// PWDN
//
// 3.3V
// VSYNC
// HREF
// RST
// D1
// D3
// D5
// D7
// NC
//
// IMPORTANT:
// This module has its own oscillator.
// There is NO XCLK connection.
//
// ============================================================

#define SPEAKER_PIN        12

#define CAM_PIN_SIOD       40
#define CAM_PIN_SIOC       41

#define CAM_PIN_D0          4
#define CAM_PIN_D1          5
#define CAM_PIN_D2          6
#define CAM_PIN_D3          7
#define CAM_PIN_D4          8
#define CAM_PIN_D5          9
#define CAM_PIN_D6         16
#define CAM_PIN_D7         17

#define CAM_PIN_PCLK       21
#define CAM_PIN_VSYNC      38
#define CAM_PIN_HREF       39

// Physically:
//
// RST  -> 3.3V
// PWDN -> GND
//
// Therefore the ESP32 does not control these pins.

#define CAM_PIN_RESET      -1
#define CAM_PIN_PWDN       -1

// No external XCLK.
//
// Your module has an onboard oscillator.
//
// On ESP32-S3 the camera driver uses the LCD_CAM peripheral.
//
#define CAM_PIN_XCLK       -1

// --------------------------------------------------------------
// PIXEL CLOCK DIVIDER
// --------------------------------------------------------------
//
// The camera sends one byte per pixel-clock tick. If that clock
// is noisy, the ESP32 can count a single tick twice and store
// the same byte twice -- which corrupts the picture even though
// every wire is connected and every byte value is correct.
//
// The camera library leaves this divider at 8 for VGA JPEG.
// Larger numbers run the bus slower and give a marginal clock
// line more margin, at the cost of frame rate:
//
//    8  = library default   (fastest, least margin)
//   16  = half speed
//   24  = one third speed
//
// Raise it until "Repeated-byte rate" below reads under 1%.

#define PIXEL_CLOCK_DIVIDER  16

// --------------------------------------------------------------
// REPEATED-BYTE ALARM THRESHOLD
// --------------------------------------------------------------
//
// A healthy picture is NOT 0% repeated bytes. Real compressed
// image data repeats bytes on its own, and how often depends
// entirely on the picture: a detailed scene runs low, a flat or
// very dark one runs high.
//
// Measured on this camera:
//
//   ground wire disconnected   6.5% - 11.1%   pictures unopenable
//   ground wire connected      1.4% -  2.2%   pictures open fine
//
// And for comparison, the same pictures re-encoded on a PC with
// no camera involved at all measured 2.0% and 3.0% -- so the
// numbers above are normal, not residual damage.
//
// 5% sits clear of both. Treat it as "something changed", not
// as a verdict.

#define REPEAT_RATE_ALARM  5.0f

// --------------------------------------------------------------
// LOW-LIGHT BOOST
// --------------------------------------------------------------
//
// 1 = allow automatic exposure up to 128x gain (good indoors)
// 0 = library defaults (8x ceiling)
//
// If bright scenes look grainy or washed out, set this to 0.

#define LOW_LIGHT_BOOST  0


// ============================================================
// INMP441 PIN CONFIGURATION
// ============================================================

// Deliberate volume boost applied AFTER the 16-bit conversion.
// 1 = no boost. Raise it if recordings are quiet, lower it if
// the diagnostic reports clipping.
#define MIC_GAIN            4

// Which half of the I2S frame to read.
//
// Both halves measured completely silent, so the channel
// question is settled and this is back to reading ONE half.
// That also halves the load this puts on the shared transfer
// hardware, which the camera depends on too.
//
// Swap to I2S_STD_SLOT_RIGHT if you ever want to re-test it.
#define MIC_SLOT            I2S_STD_SLOT_LEFT

#define MIC_BCLK            1
#define MIC_WS              2
#define MIC_DATA           42

// L/R on INMP441:
// GND = LEFT
// 3.3V = RIGHT
//
// We are using LEFT.
#define MIC_LR              GND


// ============================================================
// AUDIO CONFIGURATION
// ============================================================

#define AUDIO_SAMPLE_RATE       16000

// We capture 32-bit I2S samples from the INMP441,
// then convert them to 16-bit PCM WAV.
#define I2S_BITS                I2S_DATA_BIT_WIDTH_32BIT


// ============================================================
// GLOBAL OBJECTS
// ============================================================

WebServer server(80);

I2SClass I2S;

// A network you type in from your phone is remembered here, on the
// chip itself. It survives a power cut and a reflash of this sketch.
Preferences wifiStore;

// The network name the node broadcasts when nothing else answers.
const char* SETUP_AP_NAME = "THEO-Setup";


// ============================================================
// STATE
// ============================================================

bool cameraReady = false;
bool microphoneReady = false;

// True once the node has given up finding a network and raised its own
// so you can configure it from a phone. While this is true, NOTHING is
// allowed to touch the Wi-Fi hardware -- see loop().
bool setupModeActive = false;

// THE SPEAKER'S RESTING STATE.
//
// Idle PWM can carry supply noise into the amplifier. Fade it down after
// idle playback, and BEFORE recording: recording blocks loop(), so the
// idle timeout cannot run while the microphone is being read. Hardware
// supply or amplifier noise may still need separate diagnosis.
bool speakerAwake = false;
unsigned long speakerLastSound = 0;
const unsigned long SPEAKER_REST_MS = 1500;   // this long with no sound -> fade out
const unsigned long SPEAKER_FADE_MS = 40;     // how long each fade takes
void speakerSleep();


// ============================================================
// CAMERA INITIALIZATION
// ============================================================

bool initCamera()
{
    // No Serial.begin() here. setup() already opened the port at
    // 115200 before this runs, and calling begin() a second time
    // with a different speed makes everything printed afterwards
    // come out as garbage.

    Serial.println();
    Serial.println("Initializing OV2640...");

    // Zero-initialise: any field left unset would otherwise hold
    // whatever junk was on the stack.
    camera_config_t config = {};

    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;

    // --------------------------------------------------------
    // DATA BUS
    // --------------------------------------------------------

    config.pin_d0 = CAM_PIN_D0;
    config.pin_d1 = CAM_PIN_D1;
    config.pin_d2 = CAM_PIN_D2;
    config.pin_d3 = CAM_PIN_D3;
    config.pin_d4 = CAM_PIN_D4;
    config.pin_d5 = CAM_PIN_D5;
    config.pin_d6 = CAM_PIN_D6;
    config.pin_d7 = CAM_PIN_D7;

    // --------------------------------------------------------
    // SYNCHRONIZATION
    // --------------------------------------------------------

    config.pin_pclk  = CAM_PIN_PCLK;
    config.pin_vsync = CAM_PIN_VSYNC;
    config.pin_href  = CAM_PIN_HREF;

    // --------------------------------------------------------
    // SCCB
    // --------------------------------------------------------

    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;

    // --------------------------------------------------------
    // POWER / RESET
    // --------------------------------------------------------

    config.pin_pwdn  = CAM_PIN_PWDN;
    config.pin_reset = CAM_PIN_RESET;

    // --------------------------------------------------------
    // NO EXTERNAL XCLK
    // --------------------------------------------------------

    config.pin_xclk = CAM_PIN_XCLK;

    config.xclk_freq_hz = 20000000;

    // --------------------------------------------------------
    // IMAGE FORMAT
    // --------------------------------------------------------

    config.pixel_format = PIXFORMAT_JPEG;

    // Start conservatively.
    //
    // VGA = 640x480
    //
    // Once everything works we can increase this.
    //
    config.frame_size = FRAMESIZE_VGA;

    // JPEG quality:
    //
    // Lower = better quality / larger file
    //
    config.jpeg_quality = 12;

    // Use multiple frame buffers if PSRAM exists.
    if (psramFound())
    {
        config.fb_count = 2;
        config.fb_location = CAMERA_FB_IN_PSRAM;
        config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

        Serial.println("Camera: PSRAM detected.");
    }
    else
    {
        config.fb_count = 1;
        config.fb_location = CAMERA_FB_IN_DRAM;
        config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

        Serial.println("Camera: WARNING - no PSRAM detected.");
    }

    // --------------------------------------------------------
    // DIAGNOSTIC: memory available to the frame buffer
    // --------------------------------------------------------

    Serial.printf(
        "  PSRAM reported    : %lu bytes\n",
        (unsigned long)ESP.getPsramSize()
    );

    Serial.printf(
        "  Free heap         : %lu bytes\n",
        (unsigned long)ESP.getFreeHeap()
    );

    Serial.printf(
        "  Largest DMA block : %lu bytes\n",
        (unsigned long)heap_caps_get_largest_free_block(MALLOC_CAP_DMA)
    );

    // A VGA JPEG needs about 61,440 bytes. If the largest DMA block
    // is smaller than that and PSRAM reports 0, the frame buffer has
    // nowhere to live.

    // --------------------------------------------------------
    // INITIALIZE
    // --------------------------------------------------------

    esp_err_t result = esp_camera_init(&config);

    if (result != ESP_OK)
    {
        Serial.printf(
            "Camera initialization FAILED: 0x%x (%s)\n",
            result,
            esp_err_to_name(result)
        );

        return false;
    }

    // --------------------------------------------------------
    // SENSOR CONFIGURATION
    // --------------------------------------------------------

    sensor_t* sensor = esp_camera_sensor_get();

    if (sensor == nullptr)
    {
        Serial.println(
            "Camera sensor handle is NULL."
        );

        return false;
    }

    // --------------------------------------------------------
    // DIAGNOSTIC: which sensor actually answered
    // --------------------------------------------------------
    //
    // An OV2640 reports PID 0x26. Anything else means the driver
    // matched a different sensor than you think you have.

    Serial.printf(
        "  Sensor address    : 0x%02x\n",
        sensor->slv_addr
    );

    Serial.printf(
        "  Sensor PID        : 0x%04x  (OV2640 = 0x26)\n",
        sensor->id.PID
    );

    // --------------------------------------------------------
    // SLOW THE PIXEL CLOCK
    // --------------------------------------------------------
    //
    // Register 0xD3 in the OV2640's DSP register bank sets how
    // fast the camera clocks bytes out on the data bus. The
    // camera library writes 8 here for VGA JPEG.
    //
    // set_reg() takes the bank in bit 8 of the address, and the
    // DSP bank is 0 -- so 0x0D3 means "DSP bank, register 0xD3".
    //
    // This has to happen AFTER esp_camera_init(), because
    // changing the frame size rewrites this register.

    if (sensor->set_reg != nullptr)
    {
        sensor->set_reg(sensor, 0x0D3, 0xFF, PIXEL_CLOCK_DIVIDER);

        Serial.printf(
            "  Pixel clock div   : %d  (library default is 8)\n",
            (int)PIXEL_CLOCK_DIVIDER
        );
    }

    Serial.printf(
        "  Sensor VER        : 0x%02x   MID: 0x%02x%02x\n",
        sensor->id.VER,
        sensor->id.MIDH,
        sensor->id.MIDL
    );

    // Initial image tuning.
    sensor->set_brightness(sensor, 0);
    sensor->set_contrast(sensor, 0);
    sensor->set_saturation(sensor, 0);

    // --------------------------------------------------------
    // LOW-LIGHT HEADROOM
    // --------------------------------------------------------
    //
    // The camera library starts this sensor with its automatic
    // gain ceiling set to 8x, even though the chip supports
    // 128x. In a dim room the exposure system hits that ceiling
    // almost immediately and simply gives up, which is why
    // indoor pictures come out nearly black.
    //
    // Raising the ceiling does not force more gain -- it only
    // allows the automatic exposure to use more when the scene
    // is genuinely dark. Bright scenes are unaffected.
    //
    // Set LOW_LIGHT_BOOST to 0 to go back to library defaults.

#if LOW_LIGHT_BOOST

    if (sensor->set_gainceiling != nullptr)
    {
        sensor->set_gainceiling(sensor, GAINCEILING_128X);
    }

    if (sensor->set_aec2 != nullptr)
    {
        sensor->set_aec2(sensor, 1);
    }

    sensor->set_brightness(sensor, 1);

    Serial.println("  Low-light boost   : on  (gain ceiling 8x -> 128x)");

#else

    Serial.println("  Low-light boost   : off (library defaults)");

#endif

    Serial.println("OV2640 initialized successfully.");

    return true;
}


// ============================================================
// CAMERA SIGNAL DIAGNOSTIC
// ============================================================
//
// Watches the sync lines and the data bus for one second and
// reports what is actually moving on the wires.
//
// This is safe to run while the camera driver is active. The
// driver routes these pins to its own hardware, but the plain
// pin level is still readable at the same time. It deliberately
// uses gpio_get_level() rather than digitalRead(), because
// digitalRead() prints a warning on every single call for a pin
// owned by a peripheral -- which would flood the serial monitor
// thousands of times in this loop.
//
// Reading the result:
//
//   VSYNC edges 0      -> the sensor is not producing frames at
//                         all, or the VSYNC wire is not connected.
//   VSYNC edges 10-60  -> normal. The sensor IS streaming, and the
//                         fault is further along.
//   PCLK stuck         -> no pixel clock, so no data can arrive.
//   Data bus all stuck -> the data wires are not carrying pixels.
//
void diagnoseCameraSignals()
{
    Serial.println();
    Serial.println("Camera signal check (watching the wires for 1 second)...");

    const gpio_num_t dataPins[8] = {
        (gpio_num_t)CAM_PIN_D0, (gpio_num_t)CAM_PIN_D1,
        (gpio_num_t)CAM_PIN_D2, (gpio_num_t)CAM_PIN_D3,
        (gpio_num_t)CAM_PIN_D4, (gpio_num_t)CAM_PIN_D5,
        (gpio_num_t)CAM_PIN_D6, (gpio_num_t)CAM_PIN_D7
    };

    const gpio_num_t vsyncPin = (gpio_num_t)CAM_PIN_VSYNC;
    const gpio_num_t hrefPin  = (gpio_num_t)CAM_PIN_HREF;
    const gpio_num_t pclkPin  = (gpio_num_t)CAM_PIN_PCLK;

    uint32_t vsyncEdges = 0;
    uint32_t hrefEdges  = 0;

    bool pclkHigh = false;
    bool pclkLow  = false;

    bool dataHigh[8] = { false, false, false, false, false, false, false, false };
    bool dataLow[8]  = { false, false, false, false, false, false, false, false };

    int lastVsync = gpio_get_level(vsyncPin);
    int lastHref  = gpio_get_level(hrefPin);

    uint32_t started = millis();

    while (millis() - started < 1000)
    {
        int v = gpio_get_level(vsyncPin);

        if (v != lastVsync)
        {
            vsyncEdges++;
            lastVsync = v;
        }

        int h = gpio_get_level(hrefPin);

        if (h != lastHref)
        {
            hrefEdges++;
            lastHref = h;
        }

        if (gpio_get_level(pclkPin))
        {
            pclkHigh = true;
        }
        else
        {
            pclkLow = true;
        }

        for (uint8_t i = 0; i < 8; i++)
        {
            if (gpio_get_level(dataPins[i]))
            {
                dataHigh[i] = true;
            }
            else
            {
                dataLow[i] = true;
            }
        }
    }

    Serial.printf(
        "  VSYNC (GPIO%d) edges : %lu\n",
        CAM_PIN_VSYNC,
        (unsigned long)vsyncEdges
    );

    Serial.printf(
        "  HREF  (GPIO%d) edges : %lu\n",
        CAM_PIN_HREF,
        (unsigned long)hrefEdges
    );

    Serial.printf(
        "  PCLK  (GPIO%d)       : %s\n",
        CAM_PIN_PCLK,
        (pclkHigh && pclkLow)
            ? "toggling"
            : (pclkHigh ? "STUCK HIGH" : "STUCK LOW")
    );

    Serial.print("  Data bus D0..D7     : ");

    uint8_t stuckCount = 0;

    for (uint8_t i = 0; i < 8; i++)
    {
        if (dataHigh[i] && dataLow[i])
        {
            Serial.print('~');
        }
        else
        {
            Serial.print(dataHigh[i] ? '1' : '0');
            stuckCount++;
        }
    }

    Serial.println("   (~ = moving, 0 or 1 = stuck)");

    // Name any stuck line outright rather than making anyone
    // count characters. A single stuck data line is fatal to a
    // JPEG: the driver looks for the bytes FF D8 FF at the start
    // of every frame, and if one bit can never be 1 then no byte
    // can ever be FF, so every frame is thrown away and the
    // capture times out with nothing to show for it.

    if (stuckCount > 0)
    {
        for (uint8_t i = 0; i < 8; i++)
        {
            if (dataHigh[i] && dataLow[i])
            {
                continue;
            }

            Serial.printf(
                "  >> D%u (GPIO%d) IS STUCK %s -- this alone breaks capture.\n",
                (unsigned)i,
                (int)dataPins[i],
                dataHigh[i] ? "HIGH" : "LOW"
            );
        }

        Serial.println("     Check that wire and its breadboard row before");
        Serial.println("     changing anything in software.");
    }

    // --------------------------------------------------------
    // PLAIN-ENGLISH VERDICT
    // --------------------------------------------------------

    if (vsyncEdges == 0)
    {
        Serial.println("  >> VSYNC is not moving.");
        Serial.println("     The sensor is not sending frames. Check the VSYNC");
        Serial.println("     wire and that the camera has steady 3.3V.");
    }
    else
    {
        Serial.printf(
            "  >> About %lu frames per second are reaching the board.\n",
            (unsigned long)(vsyncEdges / 2)
        );

        if (!pclkHigh || !pclkLow)
        {
            Serial.println("     But the pixel clock is not moving, so no");
            Serial.println("     picture data can arrive. Check the DCLK wire.");
        }
    }

    Serial.println();
}


// ============================================================
// MICROPHONE INITIALIZATION
// ============================================================

bool initMicrophone()
{
    Serial.println();
    Serial.println("Initializing INMP441...");

    // --------------------------------------------------------
    // SET I2S PINS
    // --------------------------------------------------------

    I2S.setPins(
        MIC_BCLK,
        MIC_WS,
        -1,          // DOUT not used
        MIC_DATA,
        -1           // MCLK not used
    );

    // --------------------------------------------------------
    // START I2S
    // --------------------------------------------------------

    // Read ONE half of the I2S frame.
    //
    // The previous version read both halves to find out which
    // one the microphone uses. It answered that question --
    // neither -- and reading both costs twice the transfer
    // hardware the camera also needs, so it is not worth
    // keeping now that the answer is in.

    bool success = I2S.begin(
        I2S_MODE_STD,
        AUDIO_SAMPLE_RATE,
        I2S_BITS,
        I2S_SLOT_MODE_MONO,
        MIC_SLOT
    );

    if (!success)
    {
        Serial.println(
            "INMP441 initialization FAILED."
        );

        return false;
    }

    Serial.println(
        "INMP441 initialized successfully."
    );

    Serial.printf(
        "Sample rate: %d Hz\n",
        AUDIO_SAMPLE_RATE
    );

    Serial.println(
        "Format: 16 kHz / mono, one I2S half read"
    );

    return true;
}


// ============================================================
// MICROPHONE SIGNAL DIAGNOSTIC
// ============================================================
//
// Watches the microphone's data line for one second.
//
// Only SD is checked. BCLK and WS are outputs driven by the I2S
// hardware, and their pin level is not reliably readable back
// once the peripheral owns them -- so this deliberately reports
// nothing about them rather than reporting something misleading.
//
void diagnoseMicSignals()
{
    Serial.println();
    Serial.println("Microphone signal check (watching the data line for 1 second)...");

    const gpio_num_t sdPin = (gpio_num_t)MIC_DATA;

    bool sawHigh = false;
    bool sawLow  = false;

    uint32_t changes = 0;

    int last = gpio_get_level(sdPin);

    // PASS 1 -- the line exactly as wired.
    //
    // A line that changes level here MIGHT be driven. It might
    // also be floating and picking up its switching neighbours,
    // and those two look identical from here. That ambiguity is
    // why pass 2 exists.

    uint32_t started = millis();

    while (millis() - started < 500)
    {
        int v = gpio_get_level(sdPin);

        if (v)
        {
            sawHigh = true;
        }
        else
        {
            sawLow = true;
        }

        if (v != last)
        {
            changes++;
            last = v;
        }
    }

    // PASS 2 -- the same second, with a weak pull-down applied.
    //
    // A real driver overpowers a weak pull-down without noticing.
    // A floating line gets pinned low by it. So if the line still
    // reaches HIGH with the pull-down on, something is genuinely
    // driving it; if it goes quiet, pass 1 was crosstalk.

    gpio_pulldown_en(sdPin);

    delay(5);

    bool pulledHigh = false;

    uint32_t pulledChanges = 0;

    int pulledLast = gpio_get_level(sdPin);

    started = millis();

    while (millis() - started < 500)
    {
        int v = gpio_get_level(sdPin);

        if (v)
        {
            pulledHigh = true;
        }

        if (v != pulledLast)
        {
            pulledChanges++;
            pulledLast = v;
        }
    }

    gpio_pulldown_dis(sdPin);

    // PASS 3 -- the same second again, with a weak pull-UP.
    //
    // This is the decisive one, and it is sharper than pass 2.
    //
    // A pin with NOTHING connected to it sits solidly high under a
    // pull-up and never goes low. A pin wired to a real output gets
    // dragged low every time that output drives low, because a chip
    // output is vastly stronger than a weak internal pull-up.
    //
    // So: never goes low  -> the wire is not arriving here.
    //     goes low often   -> the wire is connected to something.

    gpio_pullup_en(sdPin);

    delay(5);

    bool upLow  = false;

    uint32_t upLowSamples = 0;
    uint32_t upSamples    = 0;

    started = millis();

    while (millis() - started < 500)
    {
        int v = gpio_get_level(sdPin);

        upSamples++;

        if (!v)
        {
            upLow = true;
            upLowSamples++;
        }
    }

    gpio_pullup_dis(sdPin);

    Serial.printf(
        "  SD (GPIO%d) as wired : %s   (%lu changes)\n",
        MIC_DATA,
        (sawHigh && sawLow) ? "changing" : (sawHigh ? "stuck high" : "stuck low"),
        (unsigned long)changes
    );

    Serial.printf(
        "  With a pull-down    : %s   (%lu changes)\n",
        pulledHigh ? "still reaches HIGH" : "held low",
        (unsigned long)pulledChanges
    );

    Serial.printf(
        "  With a pull-up      : %s   (low on %lu of %lu reads)\n",
        upLow ? "still pulled LOW" : "never goes low",
        (unsigned long)upLowSamples,
        (unsigned long)upSamples
    );

    // The pull-up verdict outranks the other two, so say it first
    // and say it plainly.

    // A working chip SHARES this line: it drives its own half of the
    // frame and lets go of the other half, so under a pull-up it
    // should read high something close to half the time.
    //
    // A handful of high reads out of hundreds of thousands is not a
    // chip sharing the line. It is a line being held down, and the
    // few highs are noise. Weigh how OFTEN, never merely whether --
    // an earlier version of this check tested "did it ever go high"
    // and called 13 reads in 355,146 a healthy chip.

    uint32_t upHighSamples =
        (upSamples >= upLowSamples) ? (upSamples - upLowSamples) : 0;

    bool heldDown =
        (upSamples > 0) && ((upHighSamples * 200UL) < upSamples);

    Serial.printf(
        "  Pull-up high share  : %lu of %lu reads\n",
        (unsigned long)upHighSamples,
        (unsigned long)upSamples
    );

    if (!upLow)
    {
        Serial.println("  >> NOTHING IS CONNECTED TO THIS PIN.");
        Serial.printf("     With a pull-up applied, GPIO%d never once read low in\n",
                      MIC_DATA);
        Serial.printf("     %lu reads. A wire attached to a working output would be\n",
                      (unsigned long)upSamples);
        Serial.println("     dragged low constantly. Check the SD wire itself -- both");
        Serial.println("     ends, and the breadboard rows it sits in.");
    }
    else if (heldDown)
    {
        Serial.println("  >> THIS PIN IS BEING HELD DOWN.");
        Serial.println("     A pull-up barely lifted it at all, and a working chip");
        Serial.println("     would let this line go high about half the time. So");
        Serial.println("     something is tying it low rather than talking on it.");
        Serial.println("     Two things do that: a solder bridge from SD to GND on");
        Serial.println("     the module's own pins, or a chip that is not powered");
        Serial.println("     and is clamping the line through its protection diode.");
        Serial.println("     Check SD-to-GND continuity -- there should be NONE --");
        Serial.println("     and measure 3V3 between the module's OWN VDD and GND");
        Serial.println("     pads rather than to the board's ground rail.");
    }
    else
    {
        Serial.println("  >> The SD wire IS connected to something that drives it,");
        Serial.println("     and that something lets go between bits the way a chip");
        Serial.println("     should. So the wire is fine. If the recording is still");
        Serial.println("     all zeros after this, suspect the microphone module.");
    }

    // A note on passes 1 and 2, because they misled us once.
    //
    // Watching a pin in a loop cannot keep up with a bus running at
    // a million edges a second, and a disconnected wire lying beside
    // a busy one picks up enough to look alive. Both of those made
    // pass 1 report a microphone that was not there. The pull-up in
    // pass 3 is the one to believe, and the recording numbers beat
    // all three -- they read the same pin in step with the clock.

    // Worth knowing: this runs at boot, when the room is quiet.
    // Hit /mic/check in a browser and tap the microphone while it
    // runs -- that guarantees there is something to detect.

    Serial.println();
}



// ============================================================
// WIFI
// ============================================================

// Every network THEO is allowed to join, tried top to bottom.
// Put the one you are on most often first; add or remove rows freely.

struct WifiNetwork
{
    const char* ssid;
    const char* password;
};

WifiNetwork WIFI_NETWORKS[] =
{
    { WIFI_SSID, WIFI_PASSWORD }
};

const int WIFI_NETWORK_COUNT =
    sizeof(WIFI_NETWORKS) / sizeof(WIFI_NETWORKS[0]);

// How long to wait on each network before moving to the next.
//
// This is the part that actually matters. The old code waited
// FOREVER, so a network that was not there meant the node never
// finished starting up -- no web server, nothing to talk to.

const unsigned long WIFI_ATTEMPT_MS = 12000;


// ------------------------------------------------------------
// WHY A JOIN FAILED -- THE RADIO'S OWN REASON
// ------------------------------------------------------------
//
// WiFi.status() turned out to be too blunt to be any use here. It
// gave the SAME answer -- 6 -- for a hotspot that was visibly on
// the air and for a network that no longer exists. So it cannot
// separate "not there" from "turned us away", which is now the
// only question left.
//
// The radio itself reports a far more precise reason every single
// time a join fails. This catches that and keeps the last one.

volatile uint8_t lastDisconnectReason = 0;


void onWiFiEvent(arduino_event_id_t event, arduino_event_info_t info)
{
    if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED)
    {
        lastDisconnectReason = info.wifi_sta_disconnected.reason;
    }
}


const char* disconnectMeaning(uint8_t reason)
{
    switch (reason)
    {
        case 201:
            return "NOT ON THE AIR -- nothing with that exact name answered";

        case 15:
        case 202:
        case 204:
            return "PASSWORD REFUSED -- it answered us, then rejected what we sent";

        case 2:
            return "PASSWORD REFUSED -- the handshake expired part way through";

        case 203:
            return "it heard us but would not let us in -- check Allow Others to Join";

        case 205:
            return "the attempt failed before it got anywhere";

        case 200:
            return "we lost sight of it mid-handshake -- it may have gone to sleep";

        case 36:
            return "WE GAVE UP WAITING -- nothing with that name answered in time";

        case 39:
            return "the attempt timed out";

        case 4:
            return "it dropped us for being idle";

        case 8:
            return "it disconnected us deliberately";

        case 0:
            return "the radio never reported a reason -- we simply ran out of time";

        default:
            return "unrecognised -- send me this number";
    }
}


// The old, blunt map. Kept only so the raw number still has a
// word beside it; the reason above is the one that decides.

const char* wifiReason(int status)
{
    switch (status)
    {
        case WL_NO_SSID_AVAIL:
            return "that name is not on the air right now";

        case WL_CONNECT_FAILED:
            return "the network REFUSED us -- nearly always a wrong password";

        case WL_CONNECTION_LOST:
            return "we got on and then lost it";

        case WL_IDLE_STATUS:
            return "still deciding -- ran out of time";

        case WL_DISCONNECTED:
            return "never got on";

        default:
            return "unrecognised";
    }
}


bool tryOneNetwork(const char* ssid, const char* password)
{
    if (ssid == nullptr || ssid[0] == 0)
    {
        return false;
    }

    Serial.print("[wifi] trying ");
    Serial.print(ssid);
    Serial.print(" ");

    WiFi.disconnect(true);

    delay(100);

    WiFi.mode(WIFI_STA);

    WiFi.setHostname(HOSTNAME);

    // Clear it immediately before the attempt, so whatever we read
    // afterwards belongs to THIS network and not the previous one.
    lastDisconnectReason = 0;

    WiFi.begin(ssid, password);

    unsigned long started = millis();

    // Unsigned subtraction, so this stays correct across the
    // millis() rollover roughly 49 days after boot.

    while (
        WiFi.status() != WL_CONNECTED &&
        (millis() - started) < WIFI_ATTEMPT_MS
    )
    {
        delay(250);

        Serial.print(".");
    }

    if (WiFi.status() == WL_CONNECTED)
    {
        Serial.println();

        Serial.print("[wifi] connected to ");
        Serial.println(ssid);

        return true;
    }

    int reason = WiFi.status();
    uint8_t why = lastDisconnectReason;

    Serial.println(" no answer");

    Serial.print("       why: ");
    Serial.print((int)why);
    Serial.print(" -- ");
    Serial.println(disconnectMeaning(why));

    Serial.print("            (status ");
    Serial.print(reason);
    Serial.print(": ");
    Serial.print(wifiReason(reason));
    Serial.println(")");

    return false;
}


bool connectAnyWiFi()
{
    // 1. Whatever you last typed in from your phone, FIRST.
    //
    //    It goes first because it is the most recent thing you asked
    //    for. It also matters for phone hotspots specifically: a phone
    //    switches its hotspot off after a minute or so if nothing
    //    joins, so every second spent trying dead networks first is a
    //    second the hotspot might not still be there.

    wifiStore.begin("theowifi", true);

    String savedSsid = wifiStore.getString("ssid", "");
    String savedPass = wifiStore.getString("pass", "");

    wifiStore.end();

    if (savedSsid.length() > 0)
    {
        // Counts only. The password itself is never printed -- not
        // one character of it. But the SHAPE of it is enough to catch
        // a password the form mangled on the way in.
        Serial.print("[wifi] saved from your phone: ");
        Serial.println(savedSsid);

        int lower = 0;
        int upper = 0;
        int digits = 0;
        int spaces = 0;
        int other = 0;

        for (unsigned int i = 0; i < savedPass.length(); i++)
        {
            char c = savedPass[i];

            if (c >= 'a' && c <= 'z')
            {
                lower++;
            }
            else if (c >= 'A' && c <= 'Z')
            {
                upper++;
            }
            else if (c >= '0' && c <= '9')
            {
                digits++;
            }
            else if (c == ' ')
            {
                spaces++;
            }
            else
            {
                other++;
            }
        }

        Serial.print("       password: ");
        Serial.print((int)savedPass.length());
        Serial.print(" characters -- ");
        Serial.print(lower);
        Serial.print(" lowercase, ");
        Serial.print(upper);
        Serial.print(" uppercase, ");
        Serial.print(digits);
        Serial.print(" digits, ");
        Serial.print(spaces);
        Serial.print(" spaces, ");
        Serial.print(other);
        Serial.println(" other");

        if (spaces > 0)
        {
            Serial.println("       >> There is a SPACE in the stored password.");
            Serial.println("          If you did not type one, the form mangled it.");
        }

        if (tryOneNetwork(savedSsid.c_str(), savedPass.c_str()))
        {
            return true;
        }
    }

    // 2. Then the networks written into this file.

    for (int i = 0; i < WIFI_NETWORK_COUNT; i++)
    {
        if (tryOneNetwork(
                WIFI_NETWORKS[i].ssid,
                WIFI_NETWORKS[i].password))
        {
            return true;
        }
    }

    // 3. One more go at the saved network.
    //
    //    A phone hotspot is not always awake the instant we ask. By
    //    now roughly twelve seconds have passed, which is enough for
    //    a phone that was dozing to have answered a wake-up. Costs
    //    nothing when the first attempt already worked.

    if (savedSsid.length() > 0)
    {
        Serial.println("[wifi] second try at the network you saved");

        if (tryOneNetwork(savedSsid.c_str(), savedPass.c_str()))
        {
            return true;
        }
    }

    Serial.println("[wifi] nothing answered");

    return false;
}


// ------------------------------------------------------------
// WHAT CAN THIS NODE ACTUALLY SEE?
// ------------------------------------------------------------
//
// "no answer" does not say whether the network was absent, named
// differently, or simply invisible to this chip. This prints the
// networks actually in range, which answers all three at once.

void listVisibleNetworks()
{
    Serial.println();
    Serial.println("[wifi] networks this node can actually see:");

    WiFi.mode(WIFI_STA);
    WiFi.disconnect(true);

    delay(100);

    int found = WiFi.scanNetworks();

    if (found <= 0)
    {
        Serial.println("  (none at all)");
    }
    else
    {
        for (int i = 0; i < found; i++)
        {
            Serial.print("  ");
            Serial.print(WiFi.SSID(i));
            Serial.print("    signal ");
            Serial.println(WiFi.RSSI(i));
        }
    }

    WiFi.scanDelete();

    Serial.println();
    Serial.println("  NOT in that list? Then this node cannot see it at all.");
    Serial.println("  On an iPhone that is almost always because the hotspot is");
    Serial.println("  running at 5 GHz, and this chip only does 2.4 GHz. Go to");
    Serial.println("  Settings > Personal Hotspot and turn ON Maximize");
    Serial.println("  Compatibility, then restart this node.");
    Serial.println();
    Serial.println("  IS in the list? Then copy the name EXACTLY as printed");
    Serial.println("  above into the setup page, apostrophes and spaces and all.");
    Serial.println();
}


// ------------------------------------------------------------
// SETUP NETWORK
// ------------------------------------------------------------
//
// When nothing answers, the node makes its OWN network so you can
// tell it about a new one without editing this file:
//
//   1. On your phone, join the Wi-Fi network  THEO-Setup
//   2. Open  http://192.168.4.1/setup  in a browser
//   3. Type the network name and password, press Save
//
// It stores them on the chip and restarts. They survive a power
// cut and a reflash, so it is once per network, from your phone,
// and never in this file again.

void startSetupAccessPoint()
{
    WiFi.mode(WIFI_AP);

    WiFi.softAP(SETUP_AP_NAME);

    setupModeActive = true;

    Serial.println();
    Serial.println("====================================");
    Serial.print("No network found. On your phone, join: ");
    Serial.println(SETUP_AP_NAME);
    Serial.println("Then open http://192.168.4.1/setup");
    Serial.println("====================================");
    Serial.println();
}


void handleSetupPage()
{
    String page =
        "<html><head><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "</head><body style='font-family:sans-serif;padding:24px;font-size:18px'>";

    page += "<h2>THEO network setup</h2>";
    page += "<p>Which network should THEO join?</p>";
    page += "<form action='/setup/save' method='POST'>";
    page += "<p>Network name<br>";
    page += "<input name='ssid' style='font-size:18px;padding:8px;width:90%'></p>";
    page += "<p>Password<br>";
    page += "<input name='pass' type='password' style='font-size:18px;padding:8px;width:90%'></p>";
    page += "<p><button style='font-size:18px;padding:10px 24px'>Save and restart</button></p>";
    page += "</form></body></html>";

    server.send(200, "text/html", page);
}


void handleSetupSave()
{
    String ssid = server.arg("ssid");
    String pass = server.arg("pass");

    // A space picked up from a copy-paste would make the name not match.
    // The password is deliberately left exactly as typed.
    ssid.trim();

    if (ssid.length() == 0)
    {
        server.send(
            400,
            "text/html",
            "<body style='font-family:sans-serif;padding:24px;font-size:18px'>"
            "<p>The network name cannot be empty. Go back and try again.</p></body>"
        );

        return;
    }

    wifiStore.begin("theowifi", false);

    wifiStore.putString("ssid", ssid);
    wifiStore.putString("pass", pass);

    wifiStore.end();

    Serial.print("[wifi] saved a new network: ");
    Serial.println(ssid);

    server.send(
        200,
        "text/html",
        "<body style='font-family:sans-serif;padding:24px;font-size:18px'>"
        "<p>Saved. THEO is restarting and will join that network.</p></body>"
    );

    delay(1000);

    ESP.restart();
}



void connectWiFi()
{
    Serial.println();
    Serial.println("Connecting to Wi-Fi...");

    // Listen for the radio's own report on why a join failed.
    // Registered once, here, before the first attempt.
    WiFi.onEvent(onWiFiEvent);

    WiFi.mode(WIFI_STA);

    WiFi.setHostname(
        HOSTNAME
    );

    if (!connectAnyWiFi())
    {
        listVisibleNetworks();

        startSetupAccessPoint();

        return;
    }

    Serial.println("Wi-Fi connected.");

    Serial.print(
        "IP address: "
    );

    Serial.println(
        WiFi.localIP()
    );

    Serial.print(
        "Hostname: http://"
    );

    Serial.print(
        HOSTNAME
    );

    Serial.println(
        ".local"
    );

    // --------------------------------------------------------
    // mDNS
    // --------------------------------------------------------

    if (MDNS.begin(HOSTNAME))
    {
        Serial.println(
            "mDNS started successfully."
        );
    }
    else
    {
        Serial.println(
            "mDNS failed."
        );
    }
}


// ============================================================
// WAV HEADER
// ============================================================

void writeWavHeader(
    uint8_t* header,
    uint32_t dataSize
)
{
    uint32_t fileSize =
        dataSize + 36;

    // RIFF
    memcpy(
        header,
        "RIFF",
        4
    );

    memcpy(
        header + 4,
        &fileSize,
        4
    );

    // WAVE
    memcpy(
        header + 8,
        "WAVE",
        4
    );

    // fmt
    memcpy(
        header + 12,
        "fmt ",
        4
    );

    uint32_t fmtSize = 16;

    memcpy(
        header + 16,
        &fmtSize,
        4
    );

    // PCM
    uint16_t audioFormat = 1;

    memcpy(
        header + 20,
        &audioFormat,
        2
    );

    // Mono
    uint16_t channels = 1;

    memcpy(
        header + 22,
        &channels,
        2
    );

    // Sample rate
    uint32_t sampleRate =
        AUDIO_SAMPLE_RATE;

    memcpy(
        header + 24,
        &sampleRate,
        4
    );

    // Byte rate
    uint32_t byteRate =
        AUDIO_SAMPLE_RATE * 2;

    memcpy(
        header + 28,
        &byteRate,
        4
    );

    // Block align
    uint16_t blockAlign = 2;

    memcpy(
        header + 32,
        &blockAlign,
        2
    );

    // 16-bit PCM
    uint16_t bitsPerSample = 16;

    memcpy(
        header + 34,
        &bitsPerSample,
        2
    );

    // data
    memcpy(
        header + 36,
        "data",
        4
    );

    memcpy(
        header + 40,
        &dataSize,
        4
    );
}


// ============================================================
// HTTP: ROOT
// ============================================================

void handleRoot()
{
    // A phone opening 192.168.4.1 should get the form, not JSON.
    if (setupModeActive)
    {
        handleSetupPage();

        return;
    }

    String json = "{";

    json += "\"device\":\"THEO-ESP32\",";
    json += "\"chip\":\"ESP32-S3\",";
    json += "\"camera\":";
    json += cameraReady ? "true" : "false";
    json += ",";
    json += "\"microphone\":";
    json += microphoneReady ? "true" : "false";
    json += ",";
    json += "\"ip\":\"";
    json += WiFi.localIP().toString();
    json += "\"";

    json += "}";

    server.send(
        200,
        "application/json",
        json
    );
}


// ============================================================
// HTTP: STATUS
// ============================================================

void handleStatus()
{
    String json = "{";

    json += "\"device\":\"THEO-ESP32\",";

    json += "\"camera\":{";
    json += "\"ready\":";
    json += cameraReady ? "true" : "false";
    json += ",";
    json += "\"resolution\":\"640x480\"";
    json += "},";

    json += "\"microphone\":{";
    json += "\"ready\":";
    json += microphoneReady ? "true" : "false";
    json += ",";
    json += "\"sample_rate\":";
    json += AUDIO_SAMPLE_RATE;
    json += ",";
    json += "\"channels\":1";
    json += "}";

    json += "}";

    server.send(
        200,
        "application/json",
        json
    );
}


// ============================================================
// HTTP: CAMERA CAPTURE
// ============================================================

void handleCameraCapture()
{
    if (!cameraReady)
    {
        server.send(
            503,
            "application/json",
            "{\"error\":\"camera_not_ready\"}"
        );

        return;
    }

    Serial.println(
        "Camera capture requested."
    );

    uint32_t captureStart = millis();

    camera_fb_t* frame =
        esp_camera_fb_get();

    uint32_t captureMs = millis() - captureStart;

    if (frame == nullptr)
    {
        Serial.printf(
            "Camera capture failed after %lu ms.\n",
            (unsigned long)captureMs
        );

        // About 4000 ms is the driver's own give-up time, and it
        // means no frame arrived at all. Much less than that means
        // something else refused before it ever waited.

        if (captureMs > 3500)
        {
            Serial.println(
                "  (that is the full wait -- no frame arrived at all)"
            );
        }

        // Look at the wires while the fault is fresh.
        diagnoseCameraSignals();

        server.send(
            500,
            "application/json",
            "{\"error\":\"capture_failed\"}"
        );

        return;
    }

    Serial.printf(
        "JPEG captured: %u bytes in %lu ms (%ux%u)\n",
        frame->len,
        (unsigned long)captureMs,
        (unsigned)frame->width,
        (unsigned)frame->height
    );

    // --------------------------------------------------------
    // DIAGNOSTIC: is the pixel clock being counted twice?
    // --------------------------------------------------------
    //
    // When the camera's clock line is noisy, the ESP32 latches
    // the same byte twice. The picture data is then correct but
    // padded with thousands of accidental repeats, which no
    // viewer can decode.
    //
    // Real picture data almost never repeats a byte back to
    // back -- roughly 1 time in 256, or 0.4%. Anything above
    // about 1% is the clock being double-counted, and the
    // number tells you how bad it is.

    if (frame->len > 1)
    {
        uint32_t repeatedBytes = 0;

        for (size_t i = 1; i < frame->len; i++)
        {
            if (frame->buf[i] == frame->buf[i - 1])
            {
                repeatedBytes++;
            }
        }

        float repeatRate =
            (100.0f * (float)repeatedBytes) / (float)(frame->len - 1);

        Serial.printf(
            "  Repeated-byte rate : %.2f%%  (%lu of %lu)  -- healthy is under 1%%\n",
            repeatRate,
            (unsigned long)repeatedBytes,
            (unsigned long)(frame->len - 1)
        );

        if (repeatRate > REPEAT_RATE_ALARM)
        {
            Serial.println(
                "  >> That is far above your normal range. Suspect the pixel"
            );
            Serial.println(
                "     clock again -- check the camera ground wire first, then"
            );
            Serial.println(
                "     raise PIXEL_CLOCK_DIVIDER."
            );
        }
    }

    // --------------------------------------------------------
    // DIAGNOSTIC: is this actually a complete picture?
    // --------------------------------------------------------
    //
    // Every JPEG starts with the bytes FF D8 FF and ends with
    // FF D9. Checking both ends separates two very different
    // problems that look identical in a browser:
    //
    //   starts right, ends wrong -> cut short at the camera
    //   starts wrong             -> not a picture at all

    if (frame->len >= 4)
    {
        bool soiOk =
            frame->buf[0] == 0xFF &&
            frame->buf[1] == 0xD8 &&
            frame->buf[2] == 0xFF;

        bool eoiOk =
            frame->buf[frame->len - 2] == 0xFF &&
            frame->buf[frame->len - 1] == 0xD9;

        Serial.printf(
            "  Starts like a JPEG : %s   (first bytes %02X %02X %02X)\n",
            soiOk ? "yes" : "NO",
            frame->buf[0],
            frame->buf[1],
            frame->buf[2]
        );

        Serial.printf(
            "  Ends like a JPEG   : %s   (last bytes %02X %02X)\n",
            eoiOk ? "yes" : "NO",
            frame->buf[frame->len - 2],
            frame->buf[frame->len - 1]
        );

        if (soiOk && !eoiOk)
        {
            Serial.println("  >> The picture is cut short -- the end marker is");
            Serial.println("     missing, so the browser has an incomplete file.");
        }
        else if (!soiOk)
        {
            Serial.println("  >> This is not a JPEG. What is coming off the sensor");
            Serial.println("     is not what the driver expects.");
        }
    }

    // --------------------------------------------------------
    // SEND JPEG
    // --------------------------------------------------------

    server.setContentLength(
        frame->len
    );

    server.send(
        200,
        "image/jpeg",
        ""
    );

    WiFiClient client =
        server.client();

    // Send the picture in pieces rather than one large push.
    //
    // Handing 32KB to the network in a single call returns as soon as
    // the data has been ACCEPTED, not once it has been SENT. The web
    // server then closes the connection, and whatever had not yet
    // gone out is lost -- which is how a 32,149 byte picture arrived
    // as a 15,857 byte one.
    //
    // Writing in packet-sized pieces with a yield between them gives
    // the network task time to actually transmit each piece before
    // the next is queued.

    const size_t CHUNK_BYTES = 1460;

    size_t written = 0;

    while (written < frame->len)
    {
        size_t piece = frame->len - written;

        if (piece > CHUNK_BYTES)
        {
            piece = CHUNK_BYTES;
        }

        size_t wrote =
            client.write(
                frame->buf + written,
                piece
            );

        if (wrote == 0)
        {
            break;
        }

        written += wrote;

        // Yield so the network task runs and pushes this piece out.
        delay(1);
    }

    // Give the last piece room to leave before the server tears the
    // connection down. This is a margin rather than a guarantee --
    // there is no call that reports "the send buffer is empty" -- but
    // the pieces above are what does the real work.
    delay(50);

    if (written != frame->len)
    {
        Serial.printf(
            "  >> Only %u of %u bytes were accepted for sending.\n",
            (unsigned)written,
            (unsigned)frame->len
        );
    }
    else
    {
        Serial.printf(
            "  Sent in %u pieces of up to %u bytes.\n",
            (unsigned)((frame->len + CHUNK_BYTES - 1) / CHUNK_BYTES),
            (unsigned)CHUNK_BYTES
        );
    }

    // --------------------------------------------------------
    // RETURN FRAME BUFFER
    // --------------------------------------------------------

    esp_camera_fb_return(
        frame
    );

    Serial.println(
        "JPEG sent."
    );
}


// ============================================================
// HTTP: CAMERA SETTINGS
// ============================================================

void handleCameraSettings()
{
    if (!cameraReady)
    {
        server.send(
            503,
            "application/json",
            "{\"error\":\"camera_not_ready\"}"
        );

        return;
    }

    sensor_t* sensor =
        esp_camera_sensor_get();

    String json = "{";

    json += "\"framesize\":";
    json += sensor->status.framesize;
    json += ",";

    json += "\"quality\":";
    json += sensor->status.quality;
    json += ",";

    json += "\"brightness\":";
    json += sensor->status.brightness;
    json += ",";

    json += "\"contrast\":";
    json += sensor->status.contrast;
    json += ",";

    json += "\"saturation\":";
    json += sensor->status.saturation;

    json += "}";

    server.send(
        200,
        "application/json",
        json
    );
}


// ============================================================
// HTTP: AUDIO RECORD
// ============================================================
//
// Example:
//
// /audio/record?duration=3000
//
// Produces:
//
// 16 kHz
// mono
// 16-bit PCM
// WAV
//
// ============================================================

void handleAudioRecord()
{
    speakerSleep();
    Serial.println("[listen] speaker PWM off before recording");

    if (!microphoneReady)
    {
        server.send(
            503,
            "application/json",
            "{\"error\":\"microphone_not_ready\"}"
        );

        return;
    }

    // --------------------------------------------------------
    // RECORDING DURATION
    // --------------------------------------------------------

    uint32_t durationMs = 3000;

    if (server.hasArg("duration"))
    {
        durationMs =
            server.arg(
                "duration"
            ).toInt();
    }

    // Safety limits:
    //
    // Minimum 100 ms
    // Maximum 10 seconds
    //
    durationMs =
        constrain(
            durationMs,
            100,
            10000
        );

    // --------------------------------------------------------
    // CALCULATE SIZE
    // --------------------------------------------------------

    uint32_t sampleCount =
        (
            AUDIO_SAMPLE_RATE *
            durationMs
        ) / 1000;

    uint32_t dataSize =
        sampleCount * 2;

    uint32_t totalSize =
        dataSize + 44;

    Serial.println();
    Serial.printf(
        "Recording %lu ms\n",
        durationMs
    );

    Serial.printf(
        "Samples: %lu\n",
        sampleCount
    );

    Serial.printf(
        "WAV size: %lu bytes\n",
        totalSize
    );

    // --------------------------------------------------------
    // SEND HTTP HEADERS
    // --------------------------------------------------------

    server.sendHeader(
        "Content-Disposition",
        "inline; filename=theo_audio.wav"
    );

    server.setContentLength(
        totalSize
    );

    server.send(
        200,
        "audio/wav",
        ""
    );

    WiFiClient client =
        server.client();

    // --------------------------------------------------------
    // WAV HEADER
    // --------------------------------------------------------

    uint8_t wavHeader[44];

    writeWavHeader(
        wavHeader,
        dataSize
    );

    client.write(
        wavHeader,
        sizeof(wavHeader)
    );

    // --------------------------------------------------------
    // AUDIO BUFFER
    // --------------------------------------------------------

    const size_t BUFFER_SAMPLES = 256;

    int32_t rawSamples[BUFFER_SAMPLES];

    int16_t pcmSamples[BUFFER_SAMPLES];

    uint32_t samplesCaptured = 0;

    // --------------------------------------------------------
    // DIAGNOSTIC COUNTERS FOR THIS RECORDING
    // --------------------------------------------------------

    int32_t  rawMin       = INT32_MAX;
    int32_t  rawMax       = INT32_MIN;
    uint32_t rawZeros     = 0;

    int32_t  pcmMin       = 32767;
    int32_t  pcmMax       = -32768;
    uint32_t clippedCount = 0;

    // --------------------------------------------------------
    // RECORD
    // --------------------------------------------------------

    while (samplesCaptured < sampleCount)
    {
        size_t samplesToRead =
            min(
                (uint32_t)BUFFER_SAMPLES,
                sampleCount - samplesCaptured
            );

        size_t bytesRead =
            I2S.readBytes(
                (char*)rawSamples,
                samplesToRead * sizeof(int32_t)
            );

        size_t samplesRead =
            bytesRead / sizeof(int32_t);

        if (samplesRead == 0)
        {
            Serial.println(
                "I2S read returned 0."
            );

            break;
        }

        for (size_t i = 0; i < samplesRead; i++)
        {
            int32_t raw = rawSamples[i];

            if (raw < rawMin) rawMin = raw;
            if (raw > rawMax) rawMax = raw;
            if (raw == 0)     rawZeros++;

            // The INMP441 gives 24 bits pushed up to the TOP of a
            // 32-bit slot: real data in bits 31..8, zeros in 7..0.
            // >> 16 takes the top 16 bits. MIC_GAIN is deliberate
            // boost applied afterwards.

            int32_t sample =
                (raw >> 16) * MIC_GAIN;

            if (sample > 32767 || sample < -32768)
            {
                clippedCount++;
            }

            sample =
                constrain(
                    sample,
                    -32768,
                    32767
                );

            if (sample < pcmMin) pcmMin = sample;
            if (sample > pcmMax) pcmMax = sample;

            pcmSamples[i] =
                (int16_t)sample;
        }

        client.write(
            (uint8_t*)pcmSamples,
            samplesRead * sizeof(int16_t)
        );

        samplesCaptured += samplesRead;
    }

    Serial.printf(
        "Recording complete: %lu samples\n",
        (unsigned long)samplesCaptured
    );

    // --------------------------------------------------------
    // MICROPHONE DIAGNOSTIC
    // --------------------------------------------------------

    if (samplesCaptured > 0)
    {
        Serial.printf(
            "  Raw 32-bit range  : %ld to %ld\n",
            (long)rawMin,
            (long)rawMax
        );

        Serial.printf(
            "  Raw exact zeros   : %lu of %lu\n",
            (unsigned long)rawZeros,
            (unsigned long)samplesCaptured
        );

        Serial.printf(
            "  16-bit PCM range  : %ld to %ld   (MIC_GAIN x%d)\n",
            (long)pcmMin,
            (long)pcmMax,
            MIC_GAIN
        );

        Serial.printf(
            "  Clipped samples   : %lu\n",
            (unsigned long)clippedCount
        );

        if (rawMin == 0 && rawMax == 0)
        {
            Serial.println("  >> Every single sample is exactly zero.");
            Serial.println("     Not drifting, not small -- exactly zero, which is what");
            Serial.println("     a line nobody is driving looks like when it is read at");
            Serial.println("     the right instants. The reading side is working: it");
            Serial.println("     asked for 48000 samples and got 48000, on time. So the");
            Serial.println("     silence is coming from the microphone, not from here.");
            Serial.println("     Measure 3V3 at the microphone itself with a meter, then");
            Serial.println("     check SD, SCK, WS and L/R for continuity.");
        }
        else if (pcmMax < 500 && pcmMin > -500)
        {
            Serial.println("  >> The microphone works, but the result is very quiet.");
            Serial.println("     Raise MIC_GAIN and record again.");
        }
        else if (clippedCount > (samplesCaptured / 20))
        {
            Serial.println("  >> More than 5% of samples are clipping. Lower MIC_GAIN.");
        }
        else
        {
            Serial.println("  >> Levels look healthy.");
        }
    }

    Serial.println();
}


// ============================================================
// HTTP: STOP THE MICROPHONE
// ============================================================
//
// Shuts the audio hardware down completely and frees what it was
// holding. This exists so the camera can be tested with the
// microphone entirely out of the picture, without reflashing:
//
//   1. try /camera/capture   -> note whether it works
//   2. hit  /mic/stop
//   3. try /camera/capture again
//
// If it only works after step 2, the two are competing for the
// same transfer hardware and that is the thing to solve.
//
void handleMicStop()
{
    if (!microphoneReady)
    {
        server.send(
            200,
            "application/json",
            "{\"microphone\":\"already stopped\"}"
        );

        return;
    }

    Serial.println();
    Serial.println("Stopping the microphone on request.");

    I2S.end();

    microphoneReady = false;

    Serial.println("Microphone stopped. The camera now has the");
    Serial.println("transfer hardware to itself -- try a capture.");
    Serial.println();

    server.send(
        200,
        "application/json",
        "{\"microphone\":\"stopped\",\"note\":\"try a camera capture now\"}"
    );
}


// ============================================================
// HTTP: RE-RUN THE MICROPHONE SIGNAL CHECK
// ============================================================
//
// The boot check runs while the room is quiet. Load this while
// tapping the microphone and there is guaranteed to be something
// to detect.
//
void handleMicCheck()
{
    if (!microphoneReady)
    {
        server.send(
            503,
            "application/json",
            "{\"error\":\"microphone_not_running\"}"
        );

        return;
    }

    diagnoseMicSignals();

    server.send(
        200,
        "application/json",
        "{\"microphone\":\"checked\",\"note\":\"read the serial monitor\"}"
    );
}


// ============================================================
// HTTP 404
// ============================================================

void handleNotFound()
{
    // In setup mode, whatever address you type gets you the form.
    // This is also what catches your phone's "is there internet?"
    // check, so the page often pops up on its own.
    if (setupModeActive)
    {
        handleSetupPage();

        return;
    }

    String message;

    message += "THEO ESP32\n\n";
    message += "Unknown endpoint:\n";
    message += server.uri();
    message += "\n\nAvailable endpoints:\n";
    message += "/";
    message += "\n/status";
    message += "\n/camera/capture";
    message += "\n/camera/settings";
    message += "\n/audio/record?duration=3000";

    server.send(
        404,
        "text/plain",
        message
    );
}


// ============================================================
// START HTTP SERVER
// ============================================================

// Slide the speaker pin from one level to another. A sudden jump is
// played by the amplifier as a pop; a 40 ms slide is not.
void speakerFade(int from, int to)
{
    int steps = (to > from) ? (to - from) : (from - to);

    if (steps == 0)
    {
        analogWrite(SPEAKER_PIN, to);
        return;
    }

    unsigned long stepUs = (SPEAKER_FADE_MS * 1000UL) / (unsigned long)steps;
    int direction = (to > from) ? 1 : -1;

    for (int level = from; level != to; level += direction)
    {
        analogWrite(SPEAKER_PIN, level);
        delayMicroseconds(stepUs);
    }

    analogWrite(SPEAKER_PIN, to);
}


// Bring the speaker up from off to its resting half-level, if it is asleep.
void speakerWake()
{
    if (speakerAwake)
    {
        return;
    }

    speakerFade(0, 128);

    speakerAwake = true;
}


// Take the speaker down to off and hold it there, if it is awake.
void speakerSleep()
{
    if (!speakerAwake)
    {
        return;
    }

    speakerFade(128, 0);

    speakerAwake = false;
}


void playBuffer(const uint8_t *buf, size_t len)
{
    unsigned long t = micros();

    for (size_t i = 0; i < len; i++) {
        analogWrite(SPEAKER_PIN, buf[i]);
        t += 62;
        while ((long)(micros() - t) < 0) { }

        // Let the rest of the board have a moment, about sixty times a
        // second. Playing a reply takes seconds, and without this the
        // networking side of the chip gets no time at all for the whole
        // of it.
        //
        // The timing above is unaffected: t advances by exactly 62
        // microseconds per sample regardless, so if a pause runs long
        // the next few samples simply catch up.
        if ((i & 0x3FF) == 0) { yield(); }
    }

    analogWrite(SPEAKER_PIN, 128);
}


void handleAudioSpeak()
{
    String body = server.arg("plain");

    Serial.printf("[speak] %u bytes\n", body.length());

    if (body.length() == 0)
    {
        Serial.println("       >> that request arrived with no sound in it");

        server.send(200, "text/plain", "nothing to say");

        return;
    }

    // ANSWER FIRST, THEN PLAY.
    //
    // This used to play the whole clip and answer afterwards. For every
    // second of playing, the computer sat holding an open connection
    // waiting for a reply this board was too busy to send.

    server.send(200, "text/plain", "speaking");

    speakerWake();

    playBuffer((const uint8_t *)body.c_str(), body.length());

    speakerLastSound = millis();
}


void startServer()
{
    server.on(
        "/",
        HTTP_GET,
        handleRoot
    );

    server.on(
        "/status",
        HTTP_GET,
        handleStatus
    );

    server.on(
        "/camera/capture",
        HTTP_GET,
        handleCameraCapture
    );

    server.on(
        "/camera/settings",
        HTTP_GET,
        handleCameraSettings
    );

    server.on(
        "/audio/record",
        HTTP_GET,
        handleAudioRecord
    );

    server.on(
        "/mic/stop",
        HTTP_GET,
        handleMicStop
    );

    server.on(
        "/mic/check",
        HTTP_GET,
        handleMicCheck
    );

    server.on(
        "/audio/speak",
        HTTP_POST,
        handleAudioSpeak
    );

    server.on(
        "/setup",
        HTTP_GET,
        handleSetupPage
    );

    server.on(
        "/setup/save",
        HTTP_POST,
        handleSetupSave
    );

    server.onNotFound(
        handleNotFound
    );

    server.begin();

    Serial.println();
    Serial.println(
        "THEO HTTP server started."
    );
}


// ============================================================
// SETUP
// ============================================================

void setup()
{
    // SET YOUR SERIAL MONITOR TO 115200.
    // This is the only place the port speed is set. Nothing else in
    // this file may call Serial.begin() again.

    Serial.begin(115200);

    delay(1500);

    // THE SPEAKER.
    //
    // Without these three lines the pin switches on and off about a
    // thousand times a second, which is a pitch you can hear - so the
    // speaker plays that tone instead of the voice. 62500 is above
    // anything a person can hear, which leaves only the part that
    // changes with the speech.
    //
    // These are the three lines you added by hand. They are in the file
    // now so you never have to add them again.
    //
    // One change to them: the first used to start the pin at 128, switching
    // at half level. It now starts at 0 - silent - and speakerWake() brings
    // it up just before the first sound. See the note beside speakerAwake.

    analogWrite(SPEAKER_PIN, 0);
    analogWriteFrequency(SPEAKER_PIN, 62500);
    analogWriteResolution(SPEAKER_PIN, 8);

    Serial.println("Audio update: 2026-09-28.2");
    Serial.println();
    Serial.println(
        "===================================="
    );
    Serial.println(
        "       THEO ESP32-S3 NODE"
    );
    Serial.println(
        "===================================="
    );

    // --------------------------------------------------------
    // CAMERA  (started BEFORE Wi-Fi, deliberately)
    // --------------------------------------------------------
    //
    // The camera driver carries a documented workaround for the
    // data-transfer engine on this chip freezing when Wi-Fi
    // starts up. Bringing the camera up first sidesteps the
    // whole interaction.

    cameraReady =
        initCamera();

    if (cameraReady)
    {
        // Look at the wires once at boot, before Wi-Fi adds any
        // noise, so there is a clean baseline to compare against.
        diagnoseCameraSignals();
    }

    // --------------------------------------------------------
    // MICROPHONE
    // --------------------------------------------------------

    microphoneReady =
        initMicrophone();

    if (microphoneReady)
    {
        diagnoseMicSignals();
    }

    // --------------------------------------------------------
    // WIFI
    // --------------------------------------------------------

    connectWiFi();

    // --------------------------------------------------------
    // HTTP
    // --------------------------------------------------------

    startServer();

    // --------------------------------------------------------
    // FINAL STATUS
    // --------------------------------------------------------

    Serial.println();
    Serial.println(
        "===================================="
    );

    Serial.printf(
        "Camera: %s\n",
        cameraReady ? "READY" : "FAILED"
    );

    Serial.printf(
        "Microphone: %s\n",
        microphoneReady ? "READY" : "FAILED"
    );

    Serial.println();

    Serial.print(
        "THEO ESP32 IP: "
    );

    Serial.println(
        WiFi.localIP()
    );

    Serial.print(
        "THEO ESP32 URL: http://"
    );

    Serial.print(
        HOSTNAME
    );

    Serial.println(
        ".local"
    );

    Serial.println(
        "===================================="
    );
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
    server.handleClient();

    // Rest the speaker once nothing has played for a moment. A reply
    // arrives in several pieces well under a second apart, so this never
    // cuts in between the pieces of one sentence.
    if (speakerAwake && (millis() - speakerLastSound) > SPEAKER_REST_MS)
    {
        speakerSleep();
    }

    // Keep Wi-Fi alive.
    //
    // The setupModeActive check is NOT optional. While the node is
    // running its own network, WiFi.status() is never WL_CONNECTED --
    // that reports the STATION connection, and in access-point mode
    // there isn't one. Without this check the line below fires on the
    // very first pass, and connectAnyWiFi() calls WiFi.disconnect()
    // and switches back to station mode, destroying the access point
    // a fraction of a second after it appears.

    if (
        !setupModeActive &&
        WiFi.status() != WL_CONNECTED
    )
    {
        Serial.println(
            "Wi-Fi connection lost."
        );

        // Try every saved network again, not just the last one.
        connectAnyWiFi();
    }

    delay(2);
}
