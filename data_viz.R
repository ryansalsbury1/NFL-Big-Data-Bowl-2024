library(tidyverse)
library(nflverse)
library(gt)

#load files from python
team_yards_under_expected <- read_csv("team_yards_under_expected.csv")
player_yards_over_expected <- read_csv("player_yards_over_expected.csv")

#load plays data from nflverse
player_info <- load_players(file_type = getOption("nflreadr.prefer", default = "rds"))
player_info <- player_info %>% filter((status == 'ACT' | status == 'RES' | display_name == 'Leonard Fournette' | display_name == 'Zach Ertz') & gsis_id != '00-0036501')

#merge data with players data to get gsis_id needed for headshots
player_yards_over_expected <- merge(player_yards_over_expected, player_info %>% select(display_name, gsis_id), by.x="player", by.y ="display_name", how='left')

player_yards_over_expected$total_yards_over_expected <- round(player_yards_over_expected$total_yards_over_expected,1)
player_yards_over_expected$avg_yards_over_expected <- round(player_yards_over_expected$avg_yards_over_expected,1)

gt_theme_538 <- function(data,...) {
  data %>%
    opt_all_caps()  %>%
    opt_table_font(
      font = list(
        google_font("Chivo"),
        default_fonts()
      )
    ) %>%
    tab_style(
      style = cell_borders(
        sides = "bottom", color = "transparent", weight = px(2)
      ),
      locations = cells_body(
        columns = TRUE,
        # This is a relatively sneaky way of changing the bottom border
        # Regardless of data size
        rows = nrow(data$`_data`)
      )
    )  %>% 
    cols_label(
      gsis_id = "",
      total_plays = "Catches",
      total_yards_over_expected = "Yards over Expected",
      avg_yards_over_expected = "Avg Yards Over Expected"
    ) %>%
    cols_align(align = c("center"), columns=c(position, total_plays, total_yards_over_expected, avg_yards_over_expected)) %>%
    tab_options(
      column_labels.background.color = "white",
      table.border.top.width = px(3),
      table.border.top.color = "transparent",
      table.border.bottom.color = "transparent",
      table.border.bottom.width = px(3),
      column_labels.border.top.width = px(3),
      column_labels.border.top.color = "transparent",
      column_labels.border.bottom.width = px(3),
      column_labels.border.bottom.color = "black",
      data_row.padding = px(3),
      source_notes.font.size = 12,
      table.font.size = 16,
      ...
    ) 
}


player_plot <- player_yards_over_expected %>% select(gsis_id, player, position, total_plays, total_yards_over_expected, avg_yards_over_expected) %>% arrange(desc(avg_yards_over_expected)) %>% gt() %>% data_color(
  columns = c(avg_yards_over_expected),
  fn = scales::col_numeric(
    palette = c("white", "#AF983F"),
    domain = NULL
  )) %>% gt_nfl_headshots("gsis_id") %>%
  gt_theme_538(table.width = px(575))


gtsave(player_plot, filename = "player_plot.png")


teams <- load_teams()


theme_538 <- function(base_size = 10, font = "Exo 2") {
  
  # Text setting
  txt <- element_text(size = base_size, colour = "black", face = "plain")
  bold_txt <- element_text(
    size = base_size + 2, colour = "black",
    family = "Exo 2", face = "bold"
  )
  large_txt <- element_text(size = base_size + 4, color = "black", face = "bold")
  
  
  theme_minimal(base_size = base_size, base_family = font) +
    theme(
      # Legend Settings
      legend.key = element_blank(),
      legend.background = element_blank(),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "vertical",
      
      # Backgrounds
      strip.background = element_blank(),
      strip.text = large_txt,
      plot.background = element_blank(),
      plot.margin = unit(c(1, 1, 1, 1), "lines"),
      
      # Axis & Titles
      text = txt,
      axis.text = txt,
      axis.ticks = element_blank(),
      axis.line = element_blank(),
      axis.title = bold_txt,
      plot.title = large_txt,
      
      # Panel
      panel.grid = element_line(colour = NULL),
      panel.grid.major = element_line(colour = "#D2D2D2"),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      panel.border = element_blank()
    )
}

showtext::showtext_auto()
showtext::showtext_opts(dpi = 300)
team_plot <- team_yards_under_expected %>%
  ggplot(aes(x = avg_yards_under_expected, y = reorder(team, avg_yards_under_expected))) +
  nflplotR::geom_nfl_logos(aes(team_abbr = team), width = 0.02, alpha = 1, hjust = ifelse(team_yards_under_expected$avg_yards_under_expected > 0, -.2, 1.2)) +
  geom_col(width = .5, aes(fill = if_else(avg_yards_under_expected >= 0, "#AF983F", "black"))) +

  scale_fill_identity(aesthetics = c("fill", "colour")) +
  theme_538() +
  theme(
    panel.grid.major.y = element_blank(),
    axis.text.y = element_blank(),
    axis.title.x = element_text(size = 7, vjust = -0.5),
    plot.title = element_text(size = 9, hjust = 0.05),
    plot.subtitle = element_text(hjust = 0.028),
    aspect.ratio = 1
  ) +
  geom_hline(yintercept = 0) +
  scale_x_continuous(breaks = c(-.7, -.5, -.3, -.1, .1, .3, .5, .7)) +
  labs(
    x = "yards saved per play",
    y = "",
    title = "Defensive yards saved per play after a completed pass"
  )

ggsave("team_plot.png", team_plot, bg="white")
