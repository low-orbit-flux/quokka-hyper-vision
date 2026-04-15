use std::sync::Arc;

use iced::{
    widget::{button, column, container, row, scrollable, text},
    ContentFit, Element, Length, Subscription, Task, Theme,
};
use iced::futures::SinkExt;
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};

fn main() -> iced::Result {
    iced::application("Camera Viewer", App::update, App::view)
        .theme(|_| Theme::Dark)
        .subscription(App::subscription)
        .window_size((1280.0, 720.0))
        .run_with(|| {
            let task = Task::perform(
                tokio::task::spawn_blocking(detect_cameras),
                |res| match res {
                    Ok(Ok(cams)) => Message::CamerasLoaded(cams),
                    Ok(Err(e)) => Message::LoadError(e),
                    Err(e) => Message::LoadError(e.to_string()),
                },
            );
            (App::default(), task)
        })
}

// ── data types ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct CameraDevice {
    index: u32,
    name: String,
}

// ── state ────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct App {
    cameras: Vec<CameraDevice>,
    selected: Option<u32>,
    frame: Option<iced::widget::image::Handle>,
    status: String,
}

#[derive(Debug, Clone)]
enum Message {
    CamerasLoaded(Vec<CameraDevice>),
    CameraSelected(u32),
    Frame(Arc<Vec<u8>>, u32, u32),
    Refresh,
    LoadError(String),
}

// ── update ───────────────────────────────────────────────────────────────────

impl App {
    fn update(&mut self, msg: Message) -> Task<Message> {
        match msg {
            Message::CamerasLoaded(cams) => {
                self.status = if cams.is_empty() {
                    "No cameras detected.".into()
                } else {
                    format!("{} camera(s) detected.", cams.len())
                };
                self.cameras = cams;
            }

            Message::CameraSelected(idx) => {
                self.selected = Some(idx);
                self.frame = None;
                let name = self
                    .cameras
                    .iter()
                    .find(|c| c.index == idx)
                    .map(|c| c.name.clone())
                    .unwrap_or_else(|| format!("Camera {idx}"));
                self.status = format!("Opening {name}…");
            }

            Message::Frame(pixels, w, h) => {
                self.frame = Some(iced::widget::image::Handle::from_rgba(
                    w,
                    h,
                    (*pixels).clone(),
                ));
                if self.status.starts_with("Opening") {
                    self.status = "Streaming.".into();
                }
            }

            Message::Refresh => {
                self.status = "Refreshing camera list…".into();
                self.cameras.clear();
                self.selected = None;
                self.frame = None;
                return Task::perform(
                    tokio::task::spawn_blocking(detect_cameras),
                    |res| match res {
                        Ok(Ok(cams)) => Message::CamerasLoaded(cams),
                        Ok(Err(e)) => Message::LoadError(e),
                        Err(e) => Message::LoadError(e.to_string()),
                    },
                );
            }

            Message::LoadError(e) => {
                self.status = format!("Error: {e}");
            }
        }
        Task::none()
    }

    // ── view ─────────────────────────────────────────────────────────────────

    fn view(&self) -> Element<'_, Message> {
        // Left panel: camera list
        // The scrollable content must NOT use Length::Fill on the scroll axis,
        // so the camera buttons and the header live inside it without Fill height.
        let mut scroll_content =
            column![text("Cameras").size(22)].spacing(8).padding(14);

        if self.cameras.is_empty() {
            scroll_content =
                scroll_content.push(text("No cameras found.").size(14));
        } else {
            for cam in &self.cameras {
                let selected = self.selected == Some(cam.index);
                scroll_content = scroll_content.push(
                    button(text(&cam.name))
                        .width(Length::Fill)
                        .on_press(Message::CameraSelected(cam.index))
                        .style(if selected {
                            button::primary
                        } else {
                            button::secondary
                        }),
                );
            }
        }

        // Refresh button lives outside the scrollable so it stays pinned at bottom.
        let left_col = column![
            scrollable(scroll_content).height(Length::Fill),
            button(text("Refresh"))
                .width(Length::Fill)
                .on_press(Message::Refresh),
        ]
        .spacing(8)
        .padding(8);

        let left_panel = container(left_col)
            .width(230)
            .height(Length::Fill)
            .style(container::bordered_box);

        // Right panel: video feed
        let feed: Element<Message> = match &self.frame {
            Some(handle) => iced::widget::image(handle.clone())
                .width(Length::Fill)
                .height(Length::Fill)
                .content_fit(ContentFit::Contain)
                .into(),

            None => container(
                text(if self.selected.is_some() {
                    "Waiting for first frame…"
                } else {
                    "← Select a camera to start streaming"
                })
                .size(18),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .into(),
        };

        let right_panel = container(feed)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(container::bordered_box);

        // Status bar
        let status_bar = container(text(&self.status).size(13))
            .width(Length::Fill)
            .padding([4, 8]);

        column![
            row![left_panel, right_panel].spacing(8).height(Length::Fill),
            status_bar,
        ]
        .spacing(6)
        .padding(8)
        .into()
    }

    // ── subscription: stream frames from the selected camera ──────────────────

    fn subscription(&self) -> Subscription<Message> {
        let Some(idx) = self.selected else {
            return Subscription::none();
        };

        Subscription::run_with_id(
            idx,
            iced::stream::channel(2, move |mut iced_tx| async move {
                // Bounded tokio channel: capture thread pushes, async side drains.
                let (frame_tx, mut frame_rx) =
                    tokio::sync::mpsc::channel::<(Vec<u8>, u32, u32)>(1);

                // Camera capture on a dedicated OS thread (blocking I/O).
                std::thread::spawn(move || {
                    let mut cam = match Camera::new(
                        CameraIndex::Index(idx),
                        RequestedFormat::new::<RgbFormat>(
                            RequestedFormatType::AbsoluteHighestFrameRate,
                        ),
                    ) {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("Failed to open camera {idx}: {e}");
                            return;
                        }
                    };

                    if let Err(e) = cam.open_stream() {
                        eprintln!("Failed to open stream for camera {idx}: {e}");
                        return;
                    }

                    loop {
                        match cam.frame() {
                            Ok(frame) => match frame.decode_image::<RgbFormat>() {
                                Ok(img) => {
                                    let w = img.width();
                                    let h = img.height();
                                    // nokhwa gives packed RGB; iced from_pixels expects RGBA.
                                    let rgba: Vec<u8> = img
                                        .into_raw()
                                        .chunks_exact(3)
                                        .flat_map(|c| [c[0], c[1], c[2], 255u8])
                                        .collect();
                                    if frame_tx.blocking_send((rgba, w, h)).is_err() {
                                        break; // receiver dropped → subscription changed
                                    }
                                }
                                Err(e) => eprintln!("Decode error: {e}"),
                            },
                            Err(e) => {
                                eprintln!("Frame error: {e}");
                                break;
                            }
                        }
                    }
                });

                // Async side: forward frames to iced.
                while let Some((pixels, w, h)) = frame_rx.recv().await {
                    let _ = iced_tx
                        .send(Message::Frame(Arc::new(pixels), w, h))
                        .await;
                }
            }),
        )
    }
}

// ── camera enumeration ────────────────────────────────────────────────────────

fn detect_cameras() -> Result<Vec<CameraDevice>, String> {
    nokhwa::query(ApiBackend::Auto)
        .map_err(|e| e.to_string())
        .map(|infos| {
            infos
                .into_iter()
                .map(|info| CameraDevice {
                    index: match info.index() {
                        CameraIndex::Index(i) => *i,
                        _ => 0,
                    },
                    name: info.human_name().to_string(),
                })
                .collect()
        })
}
